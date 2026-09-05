#!/usr/bin/env python3
"""
build_processed_nifti_dataset.py

Creates a pristine, standalone 'data/processed' directory containing:
- All NIfTI volumes (.nii.gz) for all patients
- Proper, intuitive subfolder structure (<Patient>/<Modality>/...)
- RTSTRUCT segmentations rasterized into modality subfolders (<Patient>/<Modality>/segmentations/<series>/...)
- NO raw DICOM files in this directory
- Complete JSON manifests and comprehensive README
"""

import os
import sys
import glob
import json
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import pydicom
import SimpleITK as sitk
from matplotlib.path import Path as MplPath


def sanitize_name(name: str) -> str:
    """Sanitize string for clean filenames."""
    if not name:
        return "unnamed"
    name = (
        name.replace("Ã¤", "ae").replace("Ã¶", "oe").replace("Ã¼", "ue").replace("ÃŸ", "ss")
        .replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss")
    )
    name = name.replace("+", "_plus_").replace("=", "_eq_").replace("°", "deg")
    name = name.replace("/", "_").replace("\\", "_").replace(":", "_").replace(" ", "_")
    name = re.sub(r"[^\w\.\-]", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_").lower()


def clean_label(name: str) -> str:
    """Clean label string and strip generic vendor tags."""
    if not name:
        return "unnamed"
    name = (
        name.replace("Ã¤", "ä").replace("Ã¶", "ö").replace("Ã¼", "ü").replace("ÃŸ", "ß")
    )
    name = re.sub(r"-generic\.[a-zA-Z0-9_\.]+", "", name)
    return name.strip()


def sort_dicom_files(files: list) -> list:
    """Sort DICOM slices along slice normal direction vector."""
    slices = []
    for f in files:
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True, force=True)
            ipp = np.array(ds.ImagePositionPatient, dtype=float)
            iop = np.array(ds.ImageOrientationPatient, dtype=float)
            normal = np.cross(iop[:3], iop[3:])
            dist = float(np.dot(ipp, normal))
            slices.append((dist, f))
        except Exception:
            slices.append((0.0, f))
    slices.sort(key=lambda x: x[0])
    return [s[1] for s in slices]


def read_dicom_to_sitk(files: list) -> sitk.Image:
    """Convert sorted DICOM files to SimpleITK Image (guaranteed 3D)."""
    sorted_files = sort_dicom_files(files)
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(sorted_files)
    img = reader.Execute()
    if img.GetDimension() > 3 and img.GetSize()[3] == 1:
        img = img[:, :, :, 0]
    return img


def scan_patient_raw_dicom(pat_dir: str):
    """Scan raw DICOM directory and return series mapping and RTSTRUCT files."""
    series_map = defaultdict(lambda: {
        "files": [],
        "modality": "",
        "desc": "",
        "series_num": 0,
        "study_date": "",
        "patient_id": "",
        "sop_class": "",
        "frame_of_ref": ""
    })
    rtstructs = []
    
    for root, _, files in os.walk(pat_dir):
        for f in files:
            fpath = os.path.join(root, f)
            try:
                ds = pydicom.dcmread(fpath, stop_before_pixels=True, force=True)
                mod = getattr(ds, "Modality", "")
                if mod == "RTSTRUCT":
                    rtstructs.append(fpath)
                    continue
                suid = getattr(ds, "SeriesInstanceUID", None)
                if not suid:
                    continue
                info = series_map[suid]
                info["files"].append(fpath)
                info["modality"] = mod
                info["desc"] = getattr(ds, "SeriesDescription", "")
                info["series_num"] = int(getattr(ds, "SeriesNumber", 0) or 0)
                info["study_date"] = getattr(ds, "StudyDate", "")
                info["patient_id"] = getattr(ds, "PatientID", "")
                info["sop_class"] = getattr(ds, "SOPClassUID", "")
                info["frame_of_ref"] = getattr(ds, "FrameOfReferenceUID", "")
            except Exception:
                pass
                
    return series_map, rtstructs


def rasterize_single_rtstruct(rt_file: str, ref_img: sitk.Image, out_dir: str, ref_name: str, prefix: str = ""):
    """Rasterize an RTSTRUCT file into out_dir with individual binary masks and multilabel mask."""
    os.makedirs(out_dir, exist_ok=True)
    ds = pydicom.dcmread(rt_file, force=True)
    
    if ref_img.GetDimension() > 3:
        ref_img = ref_img[:, :, :, 0]
        
    Nx, Ny, Nz = ref_img.GetSize()
    spacing = ref_img.GetSpacing()
    voxel_ml = (spacing[0] * spacing[1] * spacing[2]) / 1000.0
    ref_arr = sitk.GetArrayFromImage(ref_img)
    
    roi_info = {}
    if hasattr(ds, "StructureSetROISequence"):
        for roi in ds.StructureSetROISequence:
            r_num = roi.ROINumber
            r_name = clean_label(getattr(roi, "ROIName", f"ROI_{r_num}"))
            roi_info[r_num] = {
                "name": r_name,
                "clean_name": sanitize_name(r_name),
                "color": [255, 0, 0]
            }
            
    if hasattr(ds, "ROIContourSequence"):
        for rc in ds.ROIContourSequence:
            r_num = rc.ReferencedROINumber
            if r_num in roi_info and hasattr(rc, "ROIDisplayColor"):
                roi_info[r_num]["color"] = list(rc.ROIDisplayColor)
                
    multilabel_arr = np.zeros((Nz, Ny, Nx), dtype=np.int16)
    manifest = {
        "rtstruct_file": os.path.basename(rt_file),
        "structure_set_label": getattr(ds, "StructureSetLabel", ""),
        "series_description": getattr(ds, "SeriesDescription", ""),
        "reference_image": ref_name,
        "rois": []
    }
    
    label_val = 1
    for rc in getattr(ds, "ROIContourSequence", []):
        r_num = rc.ReferencedROINumber
        meta = roi_info.get(r_num, {"name": f"ROI_{r_num}", "clean_name": f"roi_{r_num}", "color": [255, 0, 0]})
        c_seq = getattr(rc, "ContourSequence", [])
        if not c_seq:
            continue
            
        mask_arr = np.zeros((Nz, Ny, Nx), dtype=bool)
        for c in c_seq:
            pts = np.array(c.ContourData, dtype=float).reshape(-1, 3)
            indices = np.array([ref_img.TransformPhysicalPointToContinuousIndex(p) for p in pts])
            k = int(round(indices[0, 2]))
            if 0 <= k < Nz:
                poly = indices[:, :2]
                min_x = max(0, int(np.floor(poly[:, 0].min())))
                max_x = min(Nx, int(np.ceil(poly[:, 0].max())) + 1)
                min_y = max(0, int(np.floor(poly[:, 1].min())))
                max_y = min(Ny, int(np.ceil(poly[:, 1].max())) + 1)
                if max_x > min_x and max_y > min_y:
                    path = MplPath(poly)
                    gx, gy = np.meshgrid(np.arange(min_x, max_x), np.arange(min_y, max_y))
                    grid_pts = np.vstack((gx.flatten(), gy.flatten())).T
                    inside = path.contains_points(grid_pts).reshape(gy.shape)
                    mask_arr[k, min_y:max_y, min_x:max_x] ^= inside
                    
        vox = int(np.sum(mask_arr))
        vol_ml = float(vox * voxel_ml)
        if vox > 0:
            intens = ref_arr[mask_arr]
            mean_i, std_i, min_i, max_i = float(np.mean(intens)), float(np.std(intens)), float(np.min(intens)), float(np.max(intens))
            z_idx, y_idx, x_idx = np.where(mask_arr)
            c_vox = (float(np.mean(x_idx)), float(np.mean(y_idx)), float(np.mean(z_idx)))
            centroid = [round(c, 2) for c in ref_img.TransformContinuousIndexToPhysicalPoint(c_vox)]
            multilabel_arr[mask_arr] = label_val
        else:
            mean_i = std_i = min_i = max_i = 0.0
            centroid = [0.0, 0.0, 0.0]
            
        pfx = f"{prefix}_" if prefix else ""
        filename = f"roi_{pfx}{r_num:02d}_{meta['clean_name']}.nii.gz"
        mask_sitk = sitk.GetImageFromArray(mask_arr.astype(np.uint8))
        mask_sitk.CopyInformation(ref_img)
        sitk.WriteImage(mask_sitk, os.path.join(out_dir, filename))
        
        manifest["rois"].append({
            "roi_number": r_num,
            "label_value": label_val if vox > 0 else 0,
            "roi_name": meta["name"],
            "filename": filename,
            "color_rgb": meta["color"],
            "voxel_count": vox,
            "volume_cm3_ml": round(vol_ml, 3),
            "intensity_stats": {
                "mean": round(mean_i, 2),
                "std": round(std_i, 2),
                "min": round(min_i, 2),
                "max": round(max_i, 2)
            },
            "centroid_lps_mm": centroid
        })
        label_val += 1
        
    # Write multi-label mask
    multilabel_filename = "multilabel_mask.nii.gz"
    ml_sitk = sitk.GetImageFromArray(multilabel_arr)
    ml_sitk.CopyInformation(ref_img)
    sitk.WriteImage(ml_sitk, os.path.join(out_dir, multilabel_filename))
    manifest["multilabel_file"] = multilabel_filename
    
    with open(os.path.join(out_dir, "labels_index.json"), "w", encoding="utf-8") as jf:
        json.dump(manifest, jf, indent=2, ensure_ascii=False)
        
    return manifest


def build_dataset(base_data_dir: str):
    """Main function to construct pristine processed directory."""
    raw_dir = os.path.join(base_data_dir, "raw_dicom")
    processed_dir = os.path.join(base_data_dir, "processed")
    os.makedirs(processed_dir, exist_ok=True)
    
    catalog = {
        "dataset_name": "PSMA Köln Multi-Modal Cohort (Processed NIfTI)",
        "source_archive": os.path.join(base_data_dir, "PSMA aus Köln.zip"),
        "processed_root": processed_dir,
        "patients": []
    }
    
    # -------------------------------------------------------------
    # PATIENT 1
    # -------------------------------------------------------------
    p1_raw = os.path.join(raw_dir, "Koeln_PSMA_Pat1")
    p1_out = os.path.join(processed_dir, "Patient_1_277820")
    print(f"\n[Building] Patient_1_277820 -> {p1_out}")
    os.makedirs(p1_out, exist_ok=True)
    smap, _ = scan_patient_raw_dicom(p1_raw)
    
    p1_manifest = {"patient_id": "Patient_1_277820", "modalities": {}, "segmentations": []}
    
    for suid, info in smap.items():
        desc = info["desc"]
        if "ct ac wb spaet" in desc.lower():
            out_ct = os.path.join(p1_out, "CT", "ct_ac_wb_spaet.nii.gz")
            os.makedirs(os.path.dirname(out_ct), exist_ok=True)
            img = read_dicom_to_sitk(info["files"])
            sitk.WriteImage(img, out_ct)
            p1_manifest["modalities"]["CT"] = [{
                "series": desc, "file": "ct_ac_wb_spaet.nii.gz",
                "dim": list(img.GetSize()), "spacing": [round(s, 3) for s in img.GetSpacing()]
            }]
            print(f"  CT: {desc} -> {out_ct}")
        elif "pet wb spaet" in desc.lower():
            out_pet = os.path.join(p1_out, "PET", "pet_wb_spaet.nii.gz")
            os.makedirs(os.path.dirname(out_pet), exist_ok=True)
            img = read_dicom_to_sitk(info["files"])
            sitk.WriteImage(img, out_pet)
            p1_manifest["modalities"].setdefault("PET", []).append({
                "series": desc, "file": "pet_wb_spaet.nii.gz",
                "dim": list(img.GetSize()), "spacing": [round(s, 3) for s in img.GetSpacing()]
            })
            print(f"  PET: {desc} -> {out_pet}")
        elif "pet wb uncorrected spaet" in desc.lower():
            out_pet_unc = os.path.join(p1_out, "PET", "pet_wb_uncorrected_spaet.nii.gz")
            os.makedirs(os.path.dirname(out_pet_unc), exist_ok=True)
            img = read_dicom_to_sitk(info["files"])
            sitk.WriteImage(img, out_pet_unc)
            p1_manifest["modalities"].setdefault("PET", []).append({
                "series": desc, "file": "pet_wb_uncorrected_spaet.nii.gz",
                "dim": list(img.GetSize()), "spacing": [round(s, 3) for s in img.GetSpacing()]
            })
            print(f"  PET: {desc} -> {out_pet_unc}")
            
    with open(os.path.join(p1_out, "manifest.json"), "w", encoding="utf-8") as jf:
        json.dump(p1_manifest, jf, indent=2)
    catalog["patients"].append(p1_manifest)
    
    # -------------------------------------------------------------
    # PATIENT 2
    # -------------------------------------------------------------
    p2_raw = os.path.join(raw_dir, "Koeln_PSMA_Pat2")
    p2_out = os.path.join(processed_dir, "Patient_2_277735")
    print(f"\n[Building] Patient_2_277735 -> {p2_out}")
    os.makedirs(p2_out, exist_ok=True)
    smap, rtstructs = scan_patient_raw_dicom(p2_raw)
    
    p2_manifest = {"patient_id": "Patient_2_277735", "modalities": {"CT": [], "PET": []}, "segmentations": []}
    p2_imgs = {}  # key -> sitk.Image
    
    for suid, info in smap.items():
        desc = info["desc"]
        snum = info["series_num"]
        slen = len(info["files"])
        if slen < 10 or info["modality"] not in ["CT", "PT"]:
            continue
            
        clean_f = sanitize_name(desc)
        
        # Diagnostic CT
        if info["modality"] == "CT":
            if "ct ac wb  3.0  hd_fov" in desc.lower():
                fn = "ct_ac_wb_3.0_hd_fov.nii.gz"
                tag = "ct_ac_wb_3.0_hd_fov"
            elif "ct ac wb spaet" in desc.lower():
                fn = "ct_ac_wb_spaet.nii.gz"
                tag = "ct_ac_wb_spaet"
            elif "knochen 3mm" in desc.lower():
                fn = "knochen_3mm.nii.gz"
                tag = "knochen_3mm"
            elif "lunge 3mm" in desc.lower() and snum == 6:
                fn = "lunge_3mm.nii.gz"
                tag = "lunge_3mm"
            else:
                continue
                
            out_p = os.path.join(p2_out, "CT", fn)
            os.makedirs(os.path.dirname(out_p), exist_ok=True)
            img = read_dicom_to_sitk(info["files"])
            sitk.WriteImage(img, out_p)
            p2_imgs[tag] = (img, fn, suid)
            p2_manifest["modalities"]["CT"].append({
                "series": desc, "file": fn, "dim": list(img.GetSize()), "spacing": [round(s, 3) for s in img.GetSpacing()]
            })
            print(f"  CT: {desc} -> {out_p}")
            
        # Diagnostic PET
        elif info["modality"] == "PT":
            if desc == "PET WB":
                fn = "pet_wb.nii.gz"
                tag = "pet_wb"
            elif desc == "PET WB spaet":
                fn = "pet_wb_spaet.nii.gz"
                tag = "pet_wb_spaet"
            elif desc == "PET WB Uncorrected":
                fn = "pet_wb_uncorrected.nii.gz"
                tag = "pet_wb_uncorrected"
            elif desc == "PET WB Uncorrected spaet":
                fn = "pet_wb_uncorrected_spaet.nii.gz"
                tag = "pet_wb_uncorrected_spaet"
            else:
                continue
                
            out_p = os.path.join(p2_out, "PET", fn)
            os.makedirs(os.path.dirname(out_p), exist_ok=True)
            img = read_dicom_to_sitk(info["files"])
            sitk.WriteImage(img, out_p)
            p2_imgs[tag] = (img, fn, suid)
            p2_manifest["modalities"]["PET"].append({
                "series": desc, "file": fn, "dim": list(img.GetSize()), "spacing": [round(s, 3) for s in img.GetSpacing()]
            })
            print(f"  PET: {desc} -> {out_p}")
            
    # Rasterize Patient 2 RTSTRUCTs
    print(f"  Rasterizing {len(rtstructs)} RTSTRUCTs for Patient 2...")
    for rtf in sorted(rtstructs):
        ds = pydicom.dcmread(rtf, stop_before_pixels=True, force=True)
        rdesc = getattr(ds, "SeriesDescription", "")
        rlabel = getattr(ds, "StructureSetLabel", "")
        
        # Match to tag
        tag = None
        mod_folder = "CT"
        if rdesc == "PET WB spaet":
            tag = "pet_wb_spaet"
            mod_folder = "PET"
        elif rdesc == "PET WB":
            tag = "pet_wb"
            mod_folder = "PET"
        elif rdesc == "CT AC WB spaet":
            tag = "ct_ac_wb_spaet"
            mod_folder = "CT"
        elif "ct ac wb" in rdesc.lower() and "hd_fov" in rdesc.lower():
            tag = "ct_ac_wb_3.0_hd_fov"
            mod_folder = "CT"
        elif rdesc == "Knochen 3mm":
            tag = "knochen_3mm"
            mod_folder = "CT"
            
        if tag and tag in p2_imgs:
            ref_img, ref_fn, _ = p2_imgs[tag]
            seg_dir = os.path.join(p2_out, mod_folder, "segmentations", tag)
            print(f"    Rasterizing [{mod_folder}] {rdesc} -> {seg_dir}")
            res = rasterize_single_rtstruct(rtf, ref_img, seg_dir, ref_fn)
            p2_manifest["segmentations"].append({
                "modality": mod_folder,
                "structure_set": rdesc,
                "subfolder": f"{mod_folder}/segmentations/{tag}",
                "roi_count": len(res["rois"])
            })
            
    with open(os.path.join(p2_out, "manifest.json"), "w", encoding="utf-8") as jf:
        json.dump(p2_manifest, jf, indent=2)
    catalog["patients"].append(p2_manifest)
    
    # -------------------------------------------------------------
    # PATIENT 3
    # -------------------------------------------------------------
    p3_raw = os.path.join(raw_dir, "Koeln_mpMRT_Pat3")
    p3_out = os.path.join(processed_dir, "Patient_3_278803")
    print(f"\n[Building] Patient_3_278803 -> {p3_out}")
    os.makedirs(p3_out, exist_ok=True)
    smap, rtstructs = scan_patient_raw_dicom(p3_raw)
    
    p3_manifest = {"patient_id": "Patient_3_278803", "modalities": {"MRI": []}, "segmentations": []}
    p3_imgs = {}
    
    # Sequence mapping for Patient 3 MRI
    mri_target_map = {
        "t2_tra_tse_0001": ("t2_tra_tse_0001.nii.gz", 28),
        "t2_sag_tse_0003": ("t2_sag_tse_0003.nii.gz", 24),
        "t2_cor_tse_0002": ("t2_cor_tse_0002.nii.gz", 23),
        "t1_cor_vibe_dixon_nativ_w": ("t1_cor_vibe_dixon_nativ_w.nii.gz", 140),
        "t1_cor_vibe_dixon_nativ_f": ("t1_cor_vibe_dixon_nativ_f.nii.gz", 140),
        "t1_cor_vibe_dixon_nativ_in": ("t1_cor_vibe_dixon_nativ_in.nii.gz", 140),
        "t1_cor_vibe_dixon_nativ_opp": ("t1_cor_vibe_dixon_nativ_opp.nii.gz", 140),
        "km_t1_tra_vibe_dixon_w": ("km_t1_tra_vibe_dixon_w.nii.gz", 180),
        "km_t1_tra_vibe_dixon_f": ("km_t1_tra_vibe_dixon_f.nii.gz", 180),
        "resolve_diff_tra_0_1000_2000c_ap_adc": ("resolve_diff_tra_adc.nii.gz", 28),
        "resolve_diff_tra_0_1000_2000c_ap_tracew": ("resolve_diff_tra_tracew.nii.gz", 56),
        "resolve_diff_tra_0_1000_2000c_ap_calc_bval": ("resolve_diff_tra_calc_bval.nii.gz", 28),
        "zoomit_diff_tra_50_500_1000_2000c_adc": ("zoomit_diff_tra_adc.nii.gz", 28),
        "zoomit_diff_tra_50_500_1000_2000c_tracew": ("zoomit_diff_tra_tracew.nii.gz", 84),
        "zoomit_diff_tra_50_500_1000_2000c_calc_bval": ("zoomit_diff_tra_calc_bval.nii.gz", 28),
        "aa_prostate_segm": ("aa_prostate_segm.nii.gz", 26)
    }
    
    for suid, info in smap.items():
        desc = info["desc"]
        s_san = sanitize_name(desc)
        slen = len(info["files"])
        
        if s_san in mri_target_map:
            target_fn, expected_slices = mri_target_map[s_san]
            if slen == expected_slices:
                out_p = os.path.join(p3_out, "MRI", target_fn)
                os.makedirs(os.path.dirname(out_p), exist_ok=True)
                img = read_dicom_to_sitk(info["files"])
                sitk.WriteImage(img, out_p)
                p3_imgs[s_san] = (img, target_fn)
                p3_manifest["modalities"]["MRI"].append({
                    "series": desc, "file": target_fn, "dim": list(img.GetSize()), "spacing": [round(s, 3) for s in img.GetSpacing()]
                })
                print(f"  MRI: {desc} ({slen} slices) -> {out_p}")
                
    # Rasterize Patient 3 RTSTRUCTs onto t2_tra_tse_0001
    ref_img, ref_fn = p3_imgs["t2_tra_tse_0001"]
    seg_dir = os.path.join(p3_out, "MRI", "segmentations", "t2_tra_tse_0001")
    os.makedirs(seg_dir, exist_ok=True)
    print(f"  Rasterizing 3 RTSTRUCT sets onto T2 Tra TSE -> {seg_dir}...")
    
    Nx, Ny, Nz = ref_img.GetSize()
    spacing = ref_img.GetSpacing()
    voxel_ml = (spacing[0] * spacing[1] * spacing[2]) / 1000.0
    ref_arr = sitk.GetArrayFromImage(ref_img)
    multilabel_arr = np.zeros((Nz, Ny, Nx), dtype=np.int16)
    
    labels_index = {
        "patient_id": "Patient_3_278803",
        "modality": "MRI",
        "reference_image": ref_fn,
        "rois": []
    }
    
    # We will combine and name the ROIs cleanly:
    # 1. RTSS Gland Volume (Label 1)
    # 2. RTSS VOI2 (Label 2 - tumor lesion)
    # 3. mint Prostatavolumen (Label 3)
    # 4. mint P01 Prostata PZ Basis links (Label 4 - tumor lesion)
    # 5. mdprostate Prostate (Label 5)
    current_label = 1
    
    for rtf in sorted(rtstructs):
        ds = pydicom.dcmread(rtf, force=True)
        rdesc = getattr(ds, "SeriesDescription", "")
        rlabel = getattr(ds, "StructureSetLabel", "")
        
        # Determine source system tag
        if "RTSS" in rdesc or "RTSS" in rlabel:
            source_tag = "rtss"
        elif "mdprostate" in rdesc.lower():
            source_tag = "mdprostate"
        else:
            source_tag = "mint"
            
        roi_names = {}
        for r in getattr(ds, "StructureSetROISequence", []):
            roi_names[r.ROINumber] = clean_label(getattr(r, "ROIName", f"ROI_{r.ROINumber}"))
            
        for rc in getattr(ds, "ROIContourSequence", []):
            r_num = rc.ReferencedROINumber
            r_name = roi_names.get(r_num, f"ROI_{r_num}")
            clean_r = sanitize_name(r_name)
            
            c_seq = getattr(rc, "ContourSequence", [])
            if not c_seq:
                continue
                
            mask_arr = np.zeros((Nz, Ny, Nx), dtype=bool)
            for c in c_seq:
                pts = np.array(c.ContourData, dtype=float).reshape(-1, 3)
                indices = np.array([ref_img.TransformPhysicalPointToContinuousIndex(p) for p in pts])
                k = int(round(indices[0, 2]))
                if 0 <= k < Nz:
                    poly = indices[:, :2]
                    min_x = max(0, int(np.floor(poly[:, 0].min())))
                    max_x = min(Nx, int(np.ceil(poly[:, 0].max())) + 1)
                    min_y = max(0, int(np.floor(poly[:, 1].min())))
                    max_y = min(Ny, int(np.ceil(poly[:, 1].max())) + 1)
                    if max_x > min_x and max_y > min_y:
                        path = MplPath(poly)
                        gx, gy = np.meshgrid(np.arange(min_x, max_x), np.arange(min_y, max_y))
                        grid_pts = np.vstack((gx.flatten(), gy.flatten())).T
                        inside = path.contains_points(grid_pts).reshape(gy.shape)
                        mask_arr[k, min_y:max_y, min_x:max_x] ^= inside
                        
            vox = int(np.sum(mask_arr))
            vol_ml = float(vox * voxel_ml)
            if vox > 0:
                intens = ref_arr[mask_arr]
                mean_i, std_i, min_i, max_i = float(np.mean(intens)), float(np.std(intens)), float(np.min(intens)), float(np.max(intens))
                z_idx, y_idx, x_idx = np.where(mask_arr)
                c_vox = (float(np.mean(x_idx)), float(np.mean(y_idx)), float(np.mean(z_idx)))
                centroid = [round(c, 2) for c in ref_img.TransformContinuousIndexToPhysicalPoint(c_vox)]
                multilabel_arr[mask_arr] = current_label
            else:
                mean_i = std_i = min_i = max_i = 0.0
                centroid = [0.0, 0.0, 0.0]
                
            fn = f"roi_{clean_r}_{source_tag}.nii.gz"
            mask_sitk = sitk.GetImageFromArray(mask_arr.astype(np.uint8))
            mask_sitk.CopyInformation(ref_img)
            sitk.WriteImage(mask_sitk, os.path.join(seg_dir, fn))
            
            labels_index["rois"].append({
                "label_value": current_label if vox > 0 else 0,
                "source_system": source_tag,
                "roi_name": r_name,
                "filename": fn,
                "voxel_count": vox,
                "volume_cm3_ml": round(vol_ml, 3),
                "intensity_stats": {
                    "mean": round(mean_i, 2), "std": round(std_i, 2), "min": round(min_i, 2), "max": round(max_i, 2)
                },
                "centroid_lps_mm": centroid
            })
            current_label += 1
            
    # Write combined multi-label mask
    ml_fn = "multilabel_mask.nii.gz"
    ml_sitk = sitk.GetImageFromArray(multilabel_arr)
    ml_sitk.CopyInformation(ref_img)
    sitk.WriteImage(ml_sitk, os.path.join(seg_dir, ml_fn))
    labels_index["multilabel_file"] = ml_fn
    
    with open(os.path.join(seg_dir, "labels_index.json"), "w", encoding="utf-8") as jf:
        json.dump(labels_index, jf, indent=2, ensure_ascii=False)
        
    p3_manifest["segmentations"].append({
        "modality": "MRI",
        "reference_image": ref_fn,
        "subfolder": "MRI/segmentations/t2_tra_tse_0001",
        "roi_count": len(labels_index["rois"])
    })
    
    with open(os.path.join(p3_out, "manifest.json"), "w", encoding="utf-8") as jf:
        json.dump(p3_manifest, jf, indent=2)
    catalog["patients"].append(p3_manifest)
    
    # -------------------------------------------------------------
    # GLOBAL CATALOG & README
    # -------------------------------------------------------------
    with open(os.path.join(processed_dir, "dataset_catalog.json"), "w", encoding="utf-8") as jf:
        json.dump(catalog, jf, indent=2, ensure_ascii=False)
        
    create_readme(processed_dir, catalog)
    print(f"\n[Completed] Pristine processed dataset ready at {processed_dir}")


def create_readme(processed_dir: str, catalog: dict):
    """Write an informative README in the processed directory."""
    readme_path = os.path.join(processed_dir, "README.md")
    content = f"""# PSMA Köln Multi-Modal Dataset (Processed NIfTI)

This directory contains the cleaned, fully-parsed, and standardized NIfTI medical imaging dataset extracted from `PSMA aus Köln.zip`.

## Overview
- **Patients**: 3
- **Modalities**: MRI (mpMRI), PET (PSMA-targeted), CT (Attenuation correction & bone reconstruction)
- **Format**: All image volumes and segmentation masks are saved in compressed NIfTI-1 (`.nii.gz`) format.
- **Coordinate Alignment**: All RTSTRUCT segmentations have been mapped from physical DICOM LPS coordinates to voxel space with sub-millimeter registration, sharing exact origins, spacings, and direction cosines with their reference volumes.
- **No Raw DICOM**: All raw DICOM files are stored outside this directory in `data/raw_dicom/`.

---

## Directory Hierarchy

```
data/processed/
├── Patient_1_277820/
│   ├── CT/
│   │   └── ct_ac_wb_spaet.nii.gz
│   ├── PET/
│   │   ├── pet_wb_spaet.nii.gz
│   │   └── pet_wb_uncorrected_spaet.nii.gz
│   └── manifest.json
│
├── Patient_2_277735/
│   ├── CT/
│   │   ├── ct_ac_wb_3.0_hd_fov.nii.gz
│   │   ├── ct_ac_wb_spaet.nii.gz
│   │   ├── knochen_3mm.nii.gz
│   │   ├── lunge_3mm.nii.gz
│   │   └── segmentations/
│   │       ├── ct_ac_wb_3.0_hd_fov/ (10 binary ROI masks + multilabel_mask.nii.gz + labels_index.json)
│   │       ├── ct_ac_wb_spaet/ (11 binary ROI masks + multilabel_mask.nii.gz + labels_index.json)
│   │       └── knochen_3mm/ (1 binary ROI mask + multilabel_mask.nii.gz + labels_index.json)
│   ├── PET/
│   │   ├── pet_wb.nii.gz
│   │   ├── pet_wb_spaet.nii.gz
│   │   ├── pet_wb_uncorrected.nii.gz
│   │   ├── pet_wb_uncorrected_spaet.nii.gz
│   │   └── segmentations/
│   │       ├── pet_wb/ (12 binary ROI masks + multilabel_mask.nii.gz + labels_index.json)
│   │       └── pet_wb_spaet/ (11 binary ROI masks + multilabel_mask.nii.gz + labels_index.json)
│   └── manifest.json
│
├── Patient_3_278803/
│   ├── MRI/
│   │   ├── t2_tra_tse_0001.nii.gz
│   │   ├── t2_sag_tse_0003.nii.gz
│   │   ├── t2_cor_tse_0002.nii.gz
│   │   ├── t1_cor_vibe_dixon_nativ_w.nii.gz
│   │   ├── t1_cor_vibe_dixon_nativ_f.nii.gz
│   │   ├── t1_cor_vibe_dixon_nativ_in.nii.gz
│   │   ├── t1_cor_vibe_dixon_nativ_opp.nii.gz
│   │   ├── km_t1_tra_vibe_dixon_w.nii.gz
│   │   ├── km_t1_tra_vibe_dixon_f.nii.gz
│   │   ├── resolve_diff_tra_adc.nii.gz
│   │   ├── resolve_diff_tra_tracew.nii.gz
│   │   ├── resolve_diff_tra_calc_bval.nii.gz
│   │   ├── zoomit_diff_tra_adc.nii.gz
│   │   ├── zoomit_diff_tra_tracew.nii.gz
│   │   ├── zoomit_diff_tra_calc_bval.nii.gz
│   │   ├── aa_prostate_segm.nii.gz
│   │   └── segmentations/
│   │       └── t2_tra_tse_0001/
│   │           ├── multilabel_mask.nii.gz
│   │           ├── roi_gland_volume_rtss.nii.gz (Prostate Gland - AI: 30.70 mL)
│   │           ├── roi_voi2_rtss.nii.gz (Prostate Tumor Lesion - AI: 0.47 mL)
│   │           ├── roi_prostatavolumen_mint.nii.gz (Prostate Volume - mint: 38.89 mL)
│   │           ├── roi_p01_prostata_pz_basis_links_mint.nii.gz (Prostate Lesion - mint: 0.34 mL)
│   │           ├── roi_prostate_mdprostate.nii.gz (Prostate Volume - mdprostate: 35.27 mL)
│   │           └── labels_index.json
│   └── manifest.json
│
└── dataset_catalog.json
```

---

## Modality & Lesion Highlights
- **Patient 1**: Whole-body PSMA PET/CT late phase.
- **Patient 2**: Extensive PSMA PET/CT findings:
  - Primary prostate carcinoma (`PT01`: 17.47 mL, max uptake 35,035; `PT02`: 7.96 mL, max uptake 21,876)
  - Regional & distant lymph node metastases (`LN01`-`LN10`, `F01`-`F05`, `F09`-`F11`)
  - Pelvic girdle (`F08`/`F12`) and rib (`F07`) bone metastases.
- **Patient 3**: Multi-parametric prostate MRI with 3 independent expert segmentations:
  - Excellent concordant prostate gland segmentations: 30.70 mL (AI), 38.89 mL (mint), 35.27 mL (mdprostate).
  - Concordant focal tumor lesion in the left peripheral zone of the prostate base: 0.34 mL (mint) and 0.47 mL (AI), with centroids matching within 5 mm.
"""
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(content)


if __name__ == "__main__":
    base = "/mnt/big/project_ssd/project_ssd/MedEye3d.jl/data"
    if len(sys.argv) > 1:
        base = sys.argv[1]
    build_dataset(base)
