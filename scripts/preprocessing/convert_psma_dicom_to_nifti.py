#!/usr/bin/env python3
"""
convert_psma_dicom_to_nifti.py

Automated pipeline to:
1. Unpack outer 'PSMA aus Köln.zip' and inner nested patient ZIPs into data/raw_dicom/
2. Scan and index all DICOM series (Modality, SeriesDescription, SOPClassUID, etc.)
3. Convert core 3D imaging series to NIfTI (.nii.gz) using SimpleITK
   - MRI sequences (T2, Dixon, DWI/ADC)
   - PET PSMA volumes (WB, WB spaet, uncorrected)
   - CT volumes (AC WB, Knochen, Lunge, Fusion)
4. Parse and rasterize RTSTRUCT segmentations into exact physical-coordinate aligned NIfTI masks:
   - Output individual binary masks per ROI
   - Output combined multi-label integer masks
   - Save segmentations under each patient's corresponding modality subfolder
5. Calculate quantitative radiomic/lesion metrics (voxel count, volume in mL, mean/min/max/std intensity)
6. Generate per-patient and cohort-level JSON manifests and a markdown analysis summary.
"""

import os
import sys
import glob
import json
import zipfile
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import pydicom
import SimpleITK as sitk
from matplotlib.path import Path as MplPath


def sanitize_filename(name: str) -> str:
    """Sanitize a string for safe use in file/folder names, handling umlauts and special chars."""
    if not name:
        return "unnamed"
    name = (
        name.replace("Ã¤", "ae")
        .replace("Ã¶", "oe")
        .replace("Ã¼", "ue")
        .replace("ÃŸ", "ss")
        .replace("ä", "ae")
        .replace("ö", "oe")
        .replace("ü", "ue")
        .replace("ß", "ss")
    )
    name = name.replace("+", "_plus_").replace("=", "_eq_").replace("°", "deg")
    name = name.replace("/", "_").replace("\\", "_").replace(":", "_").replace(" ", "_")
    name = re.sub(r"[^\w\.\-]", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_").lower()


def clean_roi_name(name: str) -> str:
    """Clean German umlauts and remove proprietary suffix strings."""
    if not name:
        return "unnamed_roi"
    cleaned = (
        name.replace("Ã¤", "ä")
        .replace("Ã¶", "ö")
        .replace("Ã¼", "ü")
        .replace("ÃŸ", "ß")
    )
    cleaned = re.sub(r"-generic\.[a-zA-Z0-9_\.]+", "", cleaned)
    return cleaned.strip()


def unzip_dataset(zip_path: str, raw_dicom_dir: str):
    """Extract outer and nested archives into raw_dicom_dir."""
    os.makedirs(raw_dicom_dir, exist_ok=True)
    print(f"[1/5] Extracting outer zip: {zip_path} -> {raw_dicom_dir}")
    
    with zipfile.ZipFile(zip_path, 'r') as outer_zip:
        outer_zip.extractall(raw_dicom_dir)
    
    patient_zip_names = ["Koeln_PSMA_Pat1.zip", "Koeln_PSMA_Pat2.zip", "Koeln_mpMRT_Pat3.zip"]
    nested_zips = [os.path.join(raw_dicom_dir, "PSMA aus Köln", pz) for pz in patient_zip_names]
    print(f"Found {len(nested_zips)} patient archives.")
    
    for nz_path in nested_zips:
        pat_name = Path(nz_path).stem
        dest_dir = os.path.join(raw_dicom_dir, pat_name)
        if not os.path.exists(dest_dir) or len(os.listdir(dest_dir)) == 0:
            print(f"  Extracting {pat_name} -> {dest_dir}...")
            os.makedirs(dest_dir, exist_ok=True)
            with zipfile.ZipFile(nz_path, 'r') as nz:
                nz.extractall(dest_dir)
        else:
            print(f"  {pat_name} already extracted, skipping.")


def scan_dicom_directory(patient_dir: str):
    """Scan all DICOM files in patient directory and group by SeriesInstanceUID."""
    series_map = defaultdict(lambda: {
        "files": [],
        "modality": "UNKNOWN",
        "series_desc": "NoDesc",
        "series_num": 0,
        "study_uid": "",
        "patient_id": "",
        "patient_name": "",
        "sop_class": "",
        "frame_of_ref": ""
    })
    
    rtstruct_files = []
    
    for root, _, files in os.walk(patient_dir):
        for f in files:
            fpath = os.path.join(root, f)
            try:
                ds = pydicom.dcmread(fpath, stop_before_pixels=True, force=True)
                modality = getattr(ds, "Modality", "UNKNOWN")
                suid = getattr(ds, "SeriesInstanceUID", None)
                if not suid:
                    continue
                
                if modality == "RTSTRUCT":
                    rtstruct_files.append(fpath)
                    continue
                
                info = series_map[suid]
                info["files"].append(fpath)
                info["modality"] = modality
                info["series_desc"] = getattr(ds, "SeriesDescription", "NoDesc")
                info["series_num"] = int(getattr(ds, "SeriesNumber", 0) or 0)
                info["study_uid"] = getattr(ds, "StudyInstanceUID", "")
                info["patient_id"] = getattr(ds, "PatientID", "")
                info["patient_name"] = str(getattr(ds, "PatientName", ""))
                info["sop_class"] = getattr(ds, "SOPClassUID", "")
                info["frame_of_ref"] = getattr(ds, "FrameOfReferenceUID", "")
            except Exception:
                pass
                
    return series_map, rtstruct_files


def convert_dicom_series_to_nifti(file_list: list, out_nii_path: str):
    """Read DICOM files using SimpleITK and write compressed NIfTI."""
    os.makedirs(os.path.dirname(out_nii_path), exist_ok=True)
    
    reader = sitk.ImageSeriesReader()
    chosen_names = sort_dicom_files_by_slice(file_list)
    reader.SetFileNames(chosen_names)
    image = reader.Execute()
    
    # If image has 4 dimensions (e.g. (X, Y, Z, 1)), squeeze to 3D
    if image.GetDimension() > 3 and image.GetSize()[3] == 1:
        image = image[:, :, :, 0]
        
    sitk.WriteImage(image, out_nii_path)
    return image


def sort_dicom_files_by_slice(file_list: list) -> list:
    """Sort DICOM files along slice normal vector."""
    slices = []
    for f in file_list:
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True, force=True)
            ipp = np.array(ds.ImagePositionPatient, dtype=float)
            iop = np.array(ds.ImageOrientationPatient, dtype=float)
            normal = np.cross(iop[:3], iop[3:])
            dist = np.dot(ipp, normal)
            slices.append((dist, f))
        except Exception:
            slices.append((0.0, f))
    slices.sort(key=lambda x: x[0])
    return [s[1] for s in slices]


def rasterize_rtstruct_contours(rtstruct_path: str, ref_image: sitk.Image, out_seg_dir: str, ref_image_name: str):
    """
    Parse an RTSTRUCT file and rasterize ROI contours onto the reference image grid.
    Exports:
    1. Individual binary masks: roi_{idx:02d}_{clean_name}.nii.gz
    2. Multi-label integer mask: {ref_image_name}_multilabel.nii.gz
    3. JSON manifest with ROI statistics.
    """
    os.makedirs(out_seg_dir, exist_ok=True)
    ds = pydicom.dcmread(rtstruct_path, force=True)
    
    # Ensure reference image is 3D
    if ref_image.GetDimension() > 3:
        ref_image = ref_image[:, :, :, 0]
        
    Nx, Ny, Nz = ref_image.GetSize()
    spacing = ref_image.GetSpacing()
    voxel_vol_ml = (spacing[0] * spacing[1] * spacing[2]) / 1000.0
    ref_arr = sitk.GetArrayFromImage(ref_image)  # shape (Nz, Ny, Nx)
    
    roi_info = {}
    if hasattr(ds, "StructureSetROISequence"):
        for roi in ds.StructureSetROISequence:
            r_num = roi.ROINumber
            r_name = clean_roi_name(getattr(roi, "ROIName", f"ROI_{r_num}"))
            roi_info[r_num] = {
                "name": r_name,
                "clean_name": sanitize_filename(r_name),
                "color": [255, 0, 0],
                "description": getattr(roi, "ROIDescription", "")
            }
            
    if hasattr(ds, "ROIContourSequence"):
        for rc in ds.ROIContourSequence:
            r_num = rc.ReferencedROINumber
            if r_num in roi_info and hasattr(rc, "ROIDisplayColor"):
                roi_info[r_num]["color"] = list(rc.ROIDisplayColor)
                
    multilabel_arr = np.zeros((Nz, Ny, Nx), dtype=np.int16)
    manifest = {
        "rtstruct_file": os.path.basename(rtstruct_path),
        "structure_set_label": getattr(ds, "StructureSetLabel", ""),
        "structure_set_date": getattr(ds, "StructureSetDate", ""),
        "series_description": getattr(ds, "SeriesDescription", ""),
        "reference_image": ref_image_name,
        "rois": []
    }
    
    roi_contours = getattr(ds, "ROIContourSequence", [])
    label_val = 1
    
    for rc in roi_contours:
        r_num = rc.ReferencedROINumber
        meta = roi_info.get(r_num, {
            "name": f"ROI_{r_num}",
            "clean_name": f"roi_{r_num}",
            "color": [255, 0, 0]
        })
        
        c_seq = getattr(rc, "ContourSequence", [])
        if not c_seq:
            continue
            
        mask_arr = np.zeros((Nz, Ny, Nx), dtype=bool)
        
        for c in c_seq:
            pts = np.array(c.ContourData, dtype=float).reshape(-1, 3)
            indices = np.array([ref_image.TransformPhysicalPointToContinuousIndex(pt) for pt in pts])
            slice_k = int(round(indices[0, 2]))
            
            if 0 <= slice_k < Nz:
                poly_xy = indices[:, :2]  # col (x), row (y)
                min_x = max(0, int(np.floor(poly_xy[:, 0].min())))
                max_x = min(Nx, int(np.ceil(poly_xy[:, 0].max())) + 1)
                min_y = max(0, int(np.floor(poly_xy[:, 1].min())))
                max_y = min(Ny, int(np.ceil(poly_xy[:, 1].max())) + 1)
                
                if max_x > min_x and max_y > min_y:
                    path = MplPath(poly_xy)
                    gx, gy = np.meshgrid(np.arange(min_x, max_x), np.arange(min_y, max_y))
                    grid_pts = np.vstack((gx.flatten(), gy.flatten())).T
                    inside = path.contains_points(grid_pts).reshape(gy.shape)
                    mask_arr[slice_k, min_y:max_y, min_x:max_x] ^= inside
                    
        voxel_count = int(np.sum(mask_arr))
        vol_ml = float(voxel_count * voxel_vol_ml)
        
        if voxel_count > 0:
            intensities = ref_arr[mask_arr]
            mean_int = float(np.mean(intensities))
            std_int = float(np.std(intensities))
            min_int = float(np.min(intensities))
            max_int = float(np.max(intensities))
            
            z_idx, y_idx, x_idx = np.where(mask_arr)
            c_vox = (float(np.mean(x_idx)), float(np.mean(y_idx)), float(np.mean(z_idx)))
            c_phys = ref_image.TransformContinuousIndexToPhysicalPoint(c_vox)
            centroid = [round(c, 2) for c in c_phys]
        else:
            mean_int = std_int = min_int = max_int = 0.0
            centroid = [0.0, 0.0, 0.0]
            
        mask_filename = f"roi_{r_num:02d}_{meta['clean_name']}.nii.gz"
        mask_path = os.path.join(out_seg_dir, mask_filename)
        mask_sitk = sitk.GetImageFromArray(mask_arr.astype(np.uint8))
        mask_sitk.CopyInformation(ref_image)
        sitk.WriteImage(mask_sitk, mask_path)
        
        if voxel_count > 0:
            multilabel_arr[mask_arr] = label_val
            
        roi_record = {
            "roi_number": r_num,
            "label_value": label_val if voxel_count > 0 else 0,
            "roi_name": meta["name"],
            "filename": mask_filename,
            "color_rgb": meta["color"],
            "voxel_count": voxel_count,
            "volume_cm3_ml": round(vol_ml, 3),
            "intensity_stats": {
                "mean": round(mean_int, 2),
                "std": round(std_int, 2),
                "min": round(min_int, 2),
                "max": round(max_int, 2)
            },
            "centroid_lps_mm": centroid
        }
        manifest["rois"].append(roi_record)
        label_val += 1
        
    multilabel_filename = f"{ref_image_name}_multilabel.nii.gz"
    multilabel_path = os.path.join(out_seg_dir, multilabel_filename)
    multilabel_sitk = sitk.GetImageFromArray(multilabel_arr)
    multilabel_sitk.CopyInformation(ref_image)
    sitk.WriteImage(multilabel_sitk, multilabel_path)
    manifest["multilabel_file"] = multilabel_filename
    
    manifest_path = os.path.join(out_seg_dir, "labels_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as jf:
        json.dump(manifest, jf, indent=2, ensure_ascii=False)
        
    return manifest


def process_cohort(data_dir: str):
    """Main workflow to unpack and convert all patients."""
    zip_path = os.path.join(data_dir, "PSMA aus Köln.zip")
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Cannot find zip file at {zip_path}")
        
    raw_dicom_dir = os.path.join(data_dir, "raw_dicom")
    patients_out_dir = os.path.join(data_dir, "patients")
    os.makedirs(patients_out_dir, exist_ok=True)
    
    # Step 1: Unzip
    unzip_dataset(zip_path, raw_dicom_dir)
    
    cohort_summary = {
        "cohort_name": "PSMA aus Köln",
        "dataset_zip": zip_path,
        "patients": []
    }
    
    patient_dirs = [
        ("Patient_1_277820", os.path.join(raw_dicom_dir, "Koeln_PSMA_Pat1")),
        ("Patient_2_277735", os.path.join(raw_dicom_dir, "Koeln_PSMA_Pat2")),
        ("Patient_3_278803", os.path.join(raw_dicom_dir, "Koeln_mpMRT_Pat3")),
    ]
    
    for pat_id, pat_raw_dir in patient_dirs:
        print(f"\n=======================================================")
        print(f"Processing {pat_id} from {pat_raw_dir}...")
        print(f"=======================================================")
        
        if not os.path.exists(pat_raw_dir):
            print(f"Directory {pat_raw_dir} does not exist, skipping.")
            continue
            
        pat_out = os.path.join(patients_out_dir, pat_id)
        series_map, rtstruct_files = scan_dicom_directory(pat_raw_dir)
        print(f"  Found {len(series_map)} image series and {len(rtstruct_files)} RTSTRUCT files.")
        
        converted_images = {}
        patient_manifest = {
            "patient_id": pat_id,
            "raw_dir": pat_raw_dir,
            "modalities": defaultdict(list),
            "segmentations": []
        }
        
        used_filenames = set()
        
        # Sort series to prioritize series with higher slice counts (real 3D volumes)
        sorted_series = sorted(series_map.items(), key=lambda x: (x[1]["modality"], -len(x[1]["files"])))
        
        # 1. Convert Image Series
        for suid, info in sorted_series:
            modality = info["modality"]
            desc = info["series_desc"]
            series_num = info["series_num"]
            slice_count = len(info["files"])
            base_clean_name = sanitize_filename(desc)
            
            # Skip dynamic 4D series (>= 1000 slices) per user instruction
            if slice_count >= 1000:
                print(f"  [Skipping dynamic/4D series] #{series_num} {desc} ({slice_count} slices)")
                continue
            # Skip 1-4 slice localizers/scouts/snapshots
            if slice_count < 5:
                print(f"  [Skipping localizer/snapshot] #{series_num} {desc} ({slice_count} slices)")
                continue
            # Skip non-anatomical secondary captures
            if modality in ["SR", "DOC", "OT"] or "reading" in desc.lower() or "printed" in desc.lower() or "snapshots" in desc.lower() or "dosisbericht" in desc.lower() or "mint lesion" in desc.lower():
                print(f"  [Skipping secondary capture/report] #{series_num} {desc}")
                continue
                
            if modality == "PT":
                mod_folder = "PET"
            elif modality == "MR":
                mod_folder = "MRI"
            elif modality == "CT":
                mod_folder = "CT"
            else:
                mod_folder = modality
                
            clean_name = base_clean_name
            target_key = f"{mod_folder}/{clean_name}"
            if target_key in used_filenames:
                clean_name = f"{base_clean_name}_s{series_num}"
                target_key = f"{mod_folder}/{clean_name}"
            used_filenames.add(target_key)
            
            out_nii_path = os.path.join(pat_out, mod_folder, f"{clean_name}.nii.gz")
            print(f"  [Converting {modality}] #{series_num} '{desc}' ({slice_count} slices) -> {out_nii_path}")
            
            try:
                sitk_img = convert_dicom_series_to_nifti(info["files"], out_nii_path)
                converted_images[suid] = {
                    "clean_name": clean_name,
                    "desc": desc,
                    "series_num": series_num,
                    "modality": mod_folder,
                    "image": sitk_img,
                    "path": out_nii_path,
                    "size": sitk_img.GetSize(),
                    "spacing": [round(s, 3) for s in sitk_img.GetSpacing()],
                    "slice_count": slice_count
                }
                patient_manifest["modalities"][mod_folder].append({
                    "series_description": desc,
                    "series_number": series_num,
                    "filename": f"{clean_name}.nii.gz",
                    "slice_count": slice_count,
                    "dimensions": list(sitk_img.GetSize()),
                    "spacing_mm": [round(s, 3) for s in sitk_img.GetSpacing()]
                })
            except Exception as e:
                print(f"    Error converting {desc}: {e}")
                
        # 2. Process RTSTRUCT Segmentations
        print(f"\n  Processing {len(rtstruct_files)} RTSTRUCT files for {pat_id}...")
        for rt_file in sorted(rtstruct_files):
            try:
                ds = pydicom.dcmread(rt_file, stop_before_pixels=True, force=True)
                rt_desc = getattr(ds, "SeriesDescription", "")
                rt_label = getattr(ds, "StructureSetLabel", "")
                print(f"  - RTSTRUCT: '{rt_desc}' (Label: '{rt_label}')")
                
                ref_suids = []
                ref_frame = getattr(ds, "ReferencedFrameOfReferenceSequence", None)
                if ref_frame:
                    for rf in ref_frame:
                        for rts in getattr(rf, "RTReferencedStudySequence", []):
                            for rseries in getattr(rts, "RTReferencedSeriesSequence", []):
                                ref_suids.append(getattr(rseries, "SeriesInstanceUID", ""))
                                
                matched_series = None
                for rsuid in ref_suids:
                    if rsuid in converted_images:
                        matched_series = converted_images[rsuid]
                        break
                        
                if not matched_series:
                    # In Patient 3, all RTSTRUCTs target T2 tra TSE
                    if pat_id == "Patient_3_278803":
                        for suid, cinfo in converted_images.items():
                            if "t2_tra" in cinfo["clean_name"] and cinfo["slice_count"] >= 20:
                                matched_series = cinfo
                                break
                    else:
                        rt_desc_lower = rt_desc.lower()
                        for suid, cinfo in converted_images.items():
                            c_desc_lower = cinfo["desc"].lower()
                            if c_desc_lower in rt_desc_lower or rt_desc_lower in c_desc_lower:
                                matched_series = cinfo
                                break
                            
                if not matched_series:
                    print(f"    WARNING: Could not find matching reference series for RTSTRUCT '{rt_desc}', skipping.")
                    continue
                    
                ref_modality = matched_series["modality"]
                ref_clean_name = matched_series["clean_name"]
                ref_img = matched_series["image"]
                
                # Naming subfolder inside <modality>/segmentations/
                if rt_desc and sanitize_filename(rt_desc) != ref_clean_name:
                    rt_subfolder_name = sanitize_filename(rt_desc)
                elif rt_label and sanitize_filename(rt_label) not in ["measurement", "rtstruct"]:
                    rt_subfolder_name = sanitize_filename(rt_label)
                else:
                    rt_subfolder_name = ref_clean_name
                    
                seg_out_dir = os.path.join(pat_out, ref_modality, "segmentations", rt_subfolder_name)
                
                print(f"    Matched to [{ref_modality}] '{matched_series['desc']}'. Rasterizing into: {seg_out_dir}")
                seg_manifest = rasterize_rtstruct_contours(rt_file, ref_img, seg_out_dir, ref_clean_name)
                patient_manifest["segmentations"].append(seg_manifest)
                
            except Exception as e:
                print(f"    Error processing RTSTRUCT {rt_file}: {e}")
                
        # Save Patient Manifest
        pat_manifest_path = os.path.join(pat_out, "patient_manifest.json")
        with open(pat_manifest_path, "w", encoding="utf-8") as pmf:
            json.dump(patient_manifest, pmf, indent=2, ensure_ascii=False)
        print(f"  Saved manifest to {pat_manifest_path}")
        
        cohort_summary["patients"].append(patient_manifest)
        
    # Save global cohort summary & markdown report
    summary_path = os.path.join(patients_out_dir, "cohort_summary.json")
    with open(summary_path, "w", encoding="utf-8") as csf:
        json.dump(cohort_summary, csf, indent=2, ensure_ascii=False)
    print(f"\nSaved cohort summary to {summary_path}")
    
    generate_markdown_report(cohort_summary, os.path.join(patients_out_dir, "cohort_analysis_report.md"))


def generate_markdown_report(summary: dict, out_md_path: str):
    """Generate a clean markdown report summarizing patients, modalities, and segmentations."""
    lines = []
    lines.append("# PSMA Köln Dataset: Processing & Cohort Analysis Report\n")
    lines.append(f"**Source Archive**: `{summary.get('dataset_zip')}`\n")
    lines.append(f"**Total Patients Processed**: {len(summary.get('patients', []))}\n")
    lines.append("\n---\n")
    
    for pat in summary.get("patients", []):
        pid = pat["patient_id"]
        lines.append(f"## Patient: {pid}\n")
        
        lines.append("### Imaging Modalities & Sequences\n")
        lines.append("| Modality | Series Description | File Name | Slices | Dimensions (X, Y, Z) | Spacing (mm) |")
        lines.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
        for mod, series_list in pat.get("modalities", {}).items():
            for s in series_list:
                dim_str = f"{s['dimensions'][0]} × {s['dimensions'][1]} × {s['dimensions'][2]}"
                sp_str = f"{s['spacing_mm'][0]} × {s['spacing_mm'][1]} × {s['spacing_mm'][2]}"
                lines.append(f"| **{mod}** | {s['series_description']} | `{s['filename']}` | {s['slice_count']} | {dim_str} | {sp_str} |")
        lines.append("")
        
        segs = pat.get("segmentations", [])
        if segs:
            lines.append("### Labeled Segmentations & Lesions (RTSTRUCT)\n")
            for seg in segs:
                lines.append(f"#### Reference: `{seg['reference_image']}` (RTSTRUCT: *{seg.get('series_description') or seg.get('structure_set_label')}*)\n")
                lines.append(f"- **Multi-label mask**: `{seg.get('multilabel_file')}`\n")
                lines.append("| ROI # | Label Val | Anatomical / Lesion Name | Volume (cm³ / mL) | Voxel Count | Mean Intensity | Max Intensity | LPS Centroid (mm) |")
                lines.append("| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :--- |")
                for roi in seg.get("rois", []):
                    c = roi.get("centroid_lps_mm", [0, 0, 0])
                    c_str = f"({c[0]}, {c[1]}, {c[2]})"
                    istat = roi.get("intensity_stats", {})
                    lines.append(f"| {roi['roi_number']} | {roi['label_value']} | **{roi['roi_name']}** | {roi['volume_cm3_ml']} | {roi['voxel_count']} | {istat.get('mean', '-')} | {istat.get('max', '-')} | {c_str} |")
                lines.append("")
        else:
            lines.append("### Segmentations\n*No RTSTRUCT segmentations present for this patient.*\n")
            
        lines.append("\n---\n")
        
    with open(out_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved markdown report to {out_md_path}")


if __name__ == "__main__":
    base_data = "/mnt/big/project_ssd/project_ssd/MedEye3d.jl/data"
    if len(sys.argv) > 1:
        base_data = sys.argv[1]
    process_cohort(base_data)
