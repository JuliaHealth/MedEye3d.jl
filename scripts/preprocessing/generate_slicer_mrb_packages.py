#!/usr/bin/env python3
"""
generate_slicer_mrb_packages.py

Constructs clinical-grade 3D Slicer MRB packages for each patient in the dataset:
- Patient 1: CT + PET fusion with Rainbow colormap & 50% opacity
- Patient 2: Multi-series CT & PET, 5 RTSTRUCT segmentations (58 ROIs) with 3D closed surfaces, and lesion markups
- Patient 3: Multi-parametric MRI (T2 3-plane, Dixon, DWI/ADC) and multi-reader prostate/lesion segmentations

Executed via 3D Slicer:
  Slicer --no-splash --no-main-window --python-script generate_slicer_mrb_packages.py
"""

import os
import sys
import json
import vtk
import slicer


def setup_slice_composites(bg_node=None, fg_node=None, fg_opacity=0.5):
    """Configure Red, Green, and Yellow slice composite nodes."""
    for color in ["Red", "Green", "Yellow"]:
        comp_node = slicer.mrmlScene.GetNodeByID(f"vtkMRMLSliceCompositeNode{color}")
        if comp_node:
            if bg_node:
                comp_node.SetBackgroundVolumeID(bg_node.GetID())
            if fg_node:
                comp_node.SetForegroundVolumeID(fg_node.GetID())
                comp_node.SetForegroundOpacity(fg_opacity)


def build_patient1(processed_dir, out_mrb):
    """Build Patient 1 MRB (PET/CT late phase)."""
    print("\n=======================================================")
    print("Building Patient 1 MRB Package...")
    print("=======================================================")
    slicer.mrmlScene.Clear(0)
    
    pat1_dir = os.path.join(processed_dir, "Patient_1_277820")
    
    # 1. CT
    ct_file = os.path.join(pat1_dir, "CT", "ct_ac_wb_spaet.nii.gz")
    ct_node = slicer.util.loadVolume(ct_file)
    ct_node.SetName("CT_AC_WB_Late")
    ct_disp = ct_node.GetDisplayNode()
    if ct_disp:
        ct_disp.SetAutoWindowLevel(False)
        ct_disp.SetWindow(350)
        ct_disp.SetLevel(40)
        
    # 2. PET
    pet_file = os.path.join(pat1_dir, "PET", "pet_wb_spaet.nii.gz")
    pet_node = slicer.util.loadVolume(pet_file)
    pet_node.SetName("PET_WB_Late")
    pet_disp = pet_node.GetDisplayNode()
    if pet_disp:
        pet_disp.SetAndObserveColorNodeID("vtkMRMLColorTableNodeRainbow")
        
    # 3. Uncorrected PET
    pet_unc_file = os.path.join(pat1_dir, "PET", "pet_wb_uncorrected_spaet.nii.gz")
    if os.path.exists(pet_unc_file):
        pet_unc = slicer.util.loadVolume(pet_unc_file)
        pet_unc.SetName("PET_WB_Late_Uncorrected")
        pud = pet_unc.GetDisplayNode()
        if pud:
            pud.SetAndObserveColorNodeID("vtkMRMLColorTableNodeRainbow")
            
    # Setup Fusion
    setup_slice_composites(bg_node=ct_node, fg_node=pet_node, fg_opacity=0.5)
    
    # Save
    os.makedirs(os.path.dirname(out_mrb), exist_ok=True)
    saved = slicer.util.saveScene(out_mrb)
    print(f"Patient 1 MRB saved ({saved}): {out_mrb} ({os.path.getsize(out_mrb)} bytes)")
    return saved


def build_patient2(processed_dir, out_mrb):
    """Build Patient 2 MRB (Whole body early/late PET/CT + 5 RTSTRUCT segmentations + markups)."""
    print("\n=======================================================")
    print("Building Patient 2 MRB Package...")
    print("=======================================================")
    slicer.mrmlScene.Clear(0)
    
    pat2_dir = os.path.join(processed_dir, "Patient_2_277735")
    
    # 1. CT Volumes
    ct_bg = slicer.util.loadVolume(os.path.join(pat2_dir, "CT", "ct_ac_wb_3.0_hd_fov.nii.gz"))
    ct_bg.SetName("CT_AC_WB_HD_FoV")
    ct_bg.GetDisplayNode().SetAutoWindowLevel(False)
    ct_bg.GetDisplayNode().SetWindow(350)
    ct_bg.GetDisplayNode().SetLevel(40)
    
    ct_late = slicer.util.loadVolume(os.path.join(pat2_dir, "CT", "ct_ac_wb_spaet.nii.gz"))
    ct_late.SetName("CT_AC_WB_Late")
    ct_late.GetDisplayNode().SetAutoWindowLevel(False)
    ct_late.GetDisplayNode().SetWindow(350)
    ct_late.GetDisplayNode().SetLevel(40)
    
    ct_bone = slicer.util.loadVolume(os.path.join(pat2_dir, "CT", "knochen_3mm.nii.gz"))
    ct_bone.SetName("CT_Knochen_3mm")
    ct_bone.GetDisplayNode().SetAutoWindowLevel(False)
    ct_bone.GetDisplayNode().SetWindow(1000)
    ct_bone.GetDisplayNode().SetLevel(400)
    
    ct_lung = slicer.util.loadVolume(os.path.join(pat2_dir, "CT", "lunge_3mm.nii.gz"))
    ct_lung.SetName("CT_Lunge_3mm")
    ct_lung.GetDisplayNode().SetAutoWindowLevel(False)
    ct_lung.GetDisplayNode().SetWindow(1400)
    ct_lung.GetDisplayNode().SetLevel(-500)
    
    # 2. PET Volumes
    pet_fg = slicer.util.loadVolume(os.path.join(pat2_dir, "PET", "pet_wb.nii.gz"))
    pet_fg.SetName("PET_WB")
    pet_fg.GetDisplayNode().SetAndObserveColorNodeID("vtkMRMLColorTableNodeRainbow")
    
    pet_late = slicer.util.loadVolume(os.path.join(pat2_dir, "PET", "pet_wb_spaet.nii.gz"))
    pet_late.SetName("PET_WB_Late")
    pet_late.GetDisplayNode().SetAndObserveColorNodeID("vtkMRMLColorTableNodeRainbow")
    
    pet_unc = slicer.util.loadVolume(os.path.join(pat2_dir, "PET", "pet_wb_uncorrected.nii.gz"))
    pet_unc.SetName("PET_WB_Uncorrected")
    pet_unc.GetDisplayNode().SetAndObserveColorNodeID("vtkMRMLColorTableNodeRainbow")
    
    pet_unc_late = slicer.util.loadVolume(os.path.join(pat2_dir, "PET", "pet_wb_uncorrected_spaet.nii.gz"))
    pet_unc_late.SetName("PET_WB_Late_Uncorrected")
    pet_unc_late.GetDisplayNode().SetAndObserveColorNodeID("vtkMRMLColorTableNodeRainbow")
    
    # Setup Fusion
    setup_slice_composites(bg_node=ct_bg, fg_node=pet_fg, fg_opacity=0.5)
    
    # 3. Load Segmentations
    seg_definitions = [
        ("PET", "pet_wb", "Segmentation_PET_WB"),
        ("PET", "pet_wb_spaet", "Segmentation_PET_WB_Late"),
        ("CT", "ct_ac_wb_3.0_hd_fov", "Segmentation_CT_AC_WB_HD_FoV"),
        ("CT", "ct_ac_wb_spaet", "Segmentation_CT_AC_WB_Late"),
        ("CT", "knochen_3mm", "Segmentation_Knochen_3mm")
    ]
    
    for mod, sub, node_name in seg_definitions:
        seg_folder = os.path.join(pat2_dir, mod, "segmentations", sub)
        ml_path = os.path.join(seg_folder, "multilabel_mask.nii.gz")
        json_path = os.path.join(seg_folder, "labels_index.json")
        
        if os.path.exists(ml_path) and os.path.exists(json_path):
            print(f"  Loading segmentation: {node_name}...")
            seg_node = slicer.util.loadSegmentation(ml_path)
            seg_node.SetName(node_name)
            segmentation = seg_node.GetSegmentation()
            
            with open(json_path, "r", encoding="utf-8") as jf:
                meta = json.load(jf)
                
            for roi in meta.get("rois", []):
                sid = f"Segment_{roi['label_value']}"
                seg = segmentation.GetSegment(sid)
                if seg:
                    seg.SetName(roi["roi_name"])
                    rgb = [c / 255.0 for c in roi.get("color_rgb", [255, 0, 0])]
                    seg.SetColor(rgb[0], rgb[1], rgb[2])
                    
            seg_node.CreateClosedSurfaceRepresentation()
            
    # 4. Markups Fiducials (Lesion Centroids)
    # Early PET lesions
    pet_meta_file = os.path.join(pat2_dir, "PET", "segmentations", "pet_wb", "labels_index.json")
    if os.path.exists(pet_meta_file):
        with open(pet_meta_file, "r", encoding="utf-8") as f:
            pet_meta = json.load(f)
            
        fid_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "Lesions_PET_WB")
        for roi in pet_meta.get("rois", []):
            c_lps = roi.get("centroid_lps_mm")
            if c_lps and c_lps != [0.0, 0.0, 0.0]:
                # Convert LPS to RAS (-x, -y, z)
                ras = (-c_lps[0], -c_lps[1], c_lps[2])
                label_text = f"{roi['roi_name'].split('-')[0]} ({roi['volume_cm3_ml']} mL, SUV {roi['intensity_stats']['max']:.0f})"
                idx = fid_node.AddControlPoint(vtk.vtkVector3d(ras[0], ras[1], ras[2]), label_text)
                
    # Save
    os.makedirs(os.path.dirname(out_mrb), exist_ok=True)
    saved = slicer.util.saveScene(out_mrb)
    print(f"Patient 2 MRB saved ({saved}): {out_mrb} ({os.path.getsize(out_mrb)} bytes)")
    return saved


def build_patient3(processed_dir, out_mrb):
    """Build Patient 3 MRB (mpMRI with multi-reader prostate and lesion segmentations)."""
    print("\n=======================================================")
    print("Building Patient 3 MRB Package...")
    print("=======================================================")
    slicer.mrmlScene.Clear(0)
    
    pat3_dir = os.path.join(processed_dir, "Patient_3_278803")
    mri_dir = os.path.join(pat3_dir, "MRI")
    
    # 1. Primary T2 Axial (Reference Volume)
    t2_tra = slicer.util.loadVolume(os.path.join(mri_dir, "t2_tra_tse_0001.nii.gz"))
    t2_tra.SetName("MRI_T2_Axial")
    
    # 2. Multi-planar T2
    t2_sag = slicer.util.loadVolume(os.path.join(mri_dir, "t2_sag_tse_0003.nii.gz"))
    t2_sag.SetName("MRI_T2_Sagittal")
    t2_cor = slicer.util.loadVolume(os.path.join(mri_dir, "t2_cor_tse_0002.nii.gz"))
    t2_cor.SetName("MRI_T2_Coronal")
    
    # 3. T1 Dixon Native Series
    for code, desc in [("w", "Water"), ("f", "Fat"), ("in", "InPhase"), ("opp", "OpposedPhase")]:
        fpath = os.path.join(mri_dir, f"t1_cor_vibe_dixon_nativ_{code}.nii.gz")
        if os.path.exists(fpath):
            vol = slicer.util.loadVolume(fpath)
            vol.SetName(f"MRI_T1_Dixon_{desc}")
            
    # 4. T1 Contrast Series
    for code, desc in [("w", "Water"), ("f", "Fat")]:
        fpath = os.path.join(mri_dir, f"km_t1_tra_vibe_dixon_{code}.nii.gz")
        if os.path.exists(fpath):
            vol = slicer.util.loadVolume(fpath)
            vol.SetName(f"MRI_T1_PostContrast_{desc}")
            
    # 5. Diffusion DWI & ADC
    for prefix in ["zoomit", "resolve"]:
        for suffix, dname in [("adc", "ADC"), ("tracew", "TraceW"), ("calc_bval", "Calc_Bval")]:
            fn = f"{prefix}_diff_tra_{suffix}.nii.gz"
            fpath = os.path.join(mri_dir, fn)
            if os.path.exists(fpath):
                vol = slicer.util.loadVolume(fpath)
                vol.SetName(f"MRI_DWI_{prefix.capitalize()}_{dname}")
                
    # 6. Anatomical alignment
    aa_path = os.path.join(mri_dir, "aa_prostate_segm.nii.gz")
    if os.path.exists(aa_path):
        aa_vol = slicer.util.loadVolume(aa_path)
        aa_vol.SetName("MRI_AA_Prostate_Segm")
        
    # Setup background
    setup_slice_composites(bg_node=t2_tra)
    
    # 7. Multi-reader Segmentation
    seg_folder = os.path.join(mri_dir, "segmentations", "t2_tra_tse_0001")
    ml_path = os.path.join(seg_folder, "multilabel_mask.nii.gz")
    json_path = os.path.join(seg_folder, "labels_index.json")
    
    if os.path.exists(ml_path) and os.path.exists(json_path):
        print("  Loading Multi-Reader Prostate Segmentation...")
        seg_node = slicer.util.loadSegmentation(ml_path)
        seg_node.SetName("Segmentation_Prostate_MultiReader")
        segmentation = seg_node.GetSegmentation()
        
        with open(json_path, "r", encoding="utf-8") as jf:
            meta = json.load(jf)
            
        color_palette = {
            1: (0.1, 0.8, 0.2),   # AI Gland - Green
            2: (0.9, 0.1, 0.1),   # AI VOI2 - Red
            3: (0.1, 0.7, 0.9),   # mint Gland - Cyan
            4: (0.95, 0.55, 0.1), # mint P01 - Orange
            5: (0.2, 0.4, 0.95)   # mdprostate Gland - Blue
        }
        
        for roi in meta.get("rois", []):
            lval = roi["label_value"]
            sid = f"Segment_{lval}"
            seg = segmentation.GetSegment(sid)
            if seg:
                seg.SetName(f"{roi['roi_name']} ({roi['source_system'].upper()})")
                c = color_palette.get(lval, (0.7, 0.7, 0.7))
                seg.SetColor(c[0], c[1], c[2])
                
        seg_node.CreateClosedSurfaceRepresentation()
        
    # 8. Markups for Tumor Focus
    fid_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", "Lesions_Prostate")
    with open(json_path, "r", encoding="utf-8") as jf:
        meta = json.load(jf)
    for roi in meta.get("rois", []):
        if roi["label_value"] in [2, 4]:  # The two lesion foci (VOI2 and P01)
            c = roi.get("centroid_lps_mm")
            if c:
                ras = (-c[0], -c[1], c[2])
                label = f"{roi['roi_name']} ({roi['source_system'].upper()}) - {roi['volume_cm3_ml']} mL"
                fid_node.AddControlPoint(vtk.vtkVector3d(ras[0], ras[1], ras[2]), label)
                
    # Save
    os.makedirs(os.path.dirname(out_mrb), exist_ok=True)
    saved = slicer.util.saveScene(out_mrb)
    print(f"Patient 3 MRB saved ({saved}): {out_mrb} ({os.path.getsize(out_mrb)} bytes)")
    return saved


def verify_mrb(mrb_path):
    """Headless verification by reloading scene and checking nodes."""
    print(f"\n[Verifying] {mrb_path}...")
    slicer.mrmlScene.Clear(0)
    loaded = slicer.util.loadScene(mrb_path)
    if not loaded:
        print(f"  ERROR: Failed to load {mrb_path}")
        return False
        
    vols = slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")
    segs = slicer.util.getNodesByClass("vtkMRMLSegmentationNode")
    fids = slicer.util.getNodesByClass("vtkMRMLMarkupsFiducialNode")
    
    print(f"  Scene loaded successfully!")
    print(f"  - Volumes: {len(vols)} ({', '.join([v.GetName() for v in vols[:4]])}...)")
    print(f"  - Segmentations: {len(segs)}")
    for s in segs:
        print(f"    * {s.GetName()}: {s.GetSegmentation().GetNumberOfSegments()} segments")
    print(f"  - Markups: {len(fids)}")
    return True


if __name__ == "__main__":
    base_dir = "/mnt/big/project_ssd/project_ssd/MedEye3d.jl/data"
    processed = os.path.join(base_dir, "processed")
    mrb_out_dir = os.path.join(base_dir, "mrb_packages")
    os.makedirs(mrb_out_dir, exist_ok=True)
    
    mrb1 = os.path.join(mrb_out_dir, "Patient_1_277820.mrb")
    mrb2 = os.path.join(mrb_out_dir, "Patient_2_277735.mrb")
    mrb3 = os.path.join(mrb_out_dir, "Patient_3_278803.mrb")
    
    # 1. Build MRBs
    build_patient1(processed, mrb1)
    build_patient2(processed, mrb2)
    build_patient3(processed, mrb3)
    
    # 2. Symlink / copy to processed/<Patient>/
    for pid, mrb_p in [("Patient_1_277820", mrb1), ("Patient_2_277735", mrb2), ("Patient_3_278803", mrb3)]:
        pat_link = os.path.join(processed, pid, f"{pid}.mrb")
        if os.path.exists(pat_link):
            os.remove(pat_link)
        os.symlink(mrb_p, pat_link)
        print(f"Linked: {pat_link} -> {mrb_p}")
        
    # 3. Verify
    ok1 = verify_mrb(mrb1)
    ok2 = verify_mrb(mrb2)
    ok3 = verify_mrb(mrb3)
    
    if ok1 and ok2 and ok3:
        print("\nALL 3 MRB PACKAGES GENERATED AND VERIFIED SUCCESSFULLY!")
        slicer.app.exit(0)
    else:
        print("\nVerification failed on one or more MRB packages.")
        slicer.app.exit(1)
