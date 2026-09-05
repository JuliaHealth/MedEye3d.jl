#!/usr/bin/env python3
"""
run_prostate_segmentation.py

Runs nnU-Net v2 ProstateZone & Lesion segmentation on Patient 3 mpMRI:
Channel 0: T2W (t2_tra_tse_0001.nii.gz)
Channel 1: ADC (zoomit_diff_tra_adc.nii.gz)
Channel 2: HBV (zoomit_diff_tra_50_500_1000_2000c_calc_bval.nii.gz, b=2000)

Deconstructs output into:
- Anatomical prostate zones (labels 1-8)
- Deep learning detected lesion (label 9)
"""

import os
import sys
import json
import tempfile
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import nibabel as nib

def resample_to_reference(moving_path, reference_img, is_label=False):
    moving_img = sitk.ReadImage(str(moving_path))
    if (moving_img.GetSize() == reference_img.GetSize() and
        moving_img.GetSpacing() == reference_img.GetSpacing() and
        moving_img.GetOrigin() == reference_img.GetOrigin() and
        moving_img.GetDirection() == reference_img.GetDirection()):
        print(f"  [Resample] {os.path.basename(moving_path)} already matches reference grid.")
        return moving_img
        
    print(f"  [Resample] Resampling {os.path.basename(moving_path)} onto reference grid...")
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_img)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkBSpline)
    resampler.SetDefaultPixelValue(0)
    return resampler.Execute(moving_img)

def main():
    base_dir = "/mnt/big/project_ssd/project_ssd/MedEye3d.jl"
    p3_proc = os.path.join(base_dir, "data", "processed", "Patient_3_278803", "MRI")
    p3_raw = os.path.join(base_dir, "data", "patients", "Patient_3_278803", "MRI")
    
    t2w_path = os.path.join(p3_proc, "t2_tra_tse_0001.nii.gz")
    adc_path = os.path.join(p3_proc, "zoomit_diff_tra_adc.nii.gz")
    hbv_path = os.path.join(p3_raw, "zoomit_diff_tra_50_500_1000_2000c_calc_bval.nii.gz")
    
    model_training_dir = os.path.join(base_dir, "models", "Dataset501_ProstateZonesLesions", "nnUNetTrainer__nnUNetPlans__3d_fullres")
    case_dir = os.path.join(base_dir, "data", "cases", "psma_patient_all_tp")
    out_anat_dir = os.path.join(case_dir, "anatomy_out_fixed_ct_3")
    os.makedirs(out_anat_dir, exist_ok=True)
    
    print("[1/5] Checking input modality files...")
    for label, path in [("T2W (Channel 0)", t2w_path), ("ADC (Channel 1)", adc_path), ("HBV (Channel 2)", hbv_path)]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {label}: {path}")
        print(f"  ✅ {label}: {path}")
        
    with tempfile.TemporaryDirectory() as tmp_in, tempfile.TemporaryDirectory() as tmp_out:
        print("\n[2/5] Aligning and preparing multi-channel input...")
        t2w_img = sitk.ReadImage(t2w_path)
        case_id = "CASE_00000"
        
        t2_out = os.path.join(tmp_in, f"{case_id}_0000.nii.gz")
        adc_out = os.path.join(tmp_in, f"{case_id}_0001.nii.gz")
        hbv_out = os.path.join(tmp_in, f"{case_id}_0002.nii.gz")
        
        sitk.WriteImage(t2w_img, t2_out)
        adc_res = resample_to_reference(adc_path, t2w_img)
        sitk.WriteImage(adc_res, adc_out)
        hbv_res = resample_to_reference(hbv_path, t2w_img)
        sitk.WriteImage(hbv_res, hbv_out)
        print("  ✅ Channels 0000 (T2W), 0001 (ADC), 0002 (HBV) prepared.")
        
        print("\n[3/5] Loading nnUNetPredictor and running inference...")
        os.environ["nnUNet_results"] = str(Path(model_training_dir).parent.parent)
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            device=device,
            verbose=False,
            verbose_preprocessing=False
        )
        # Using folds 0 1 2 3 4 for full ensemble
        folds = [0, 1, 2, 3, 4]
        print(f"  Initializing predictor with folds {folds}...")
        predictor.initialize_from_trained_model_folder(
            model_training_output_dir=model_training_dir,
            use_folds=folds,
            checkpoint_name="checkpoint_final.pth"
        )
        
        print("  Running prediction on 3D fullres...")
        predictor.predict_from_files(
            list_of_lists_or_source_folder=str(tmp_in),
            output_folder_or_list_of_truncated_output_files=str(tmp_out),
            save_probabilities=False,
            overwrite=True,
            num_processes_preprocessing=2,
            num_processes_segmentation_export=2
        )
        
        pred_file = os.path.join(tmp_out, f"{case_id}.nii.gz")
        if not os.path.exists(pred_file):
            raise RuntimeError(f"Prediction failed, {pred_file} not found!")
            
        print("\n[4/5] Analyzing predicted zones and lesion...")
        pred_obj = nib.load(pred_file)
        pred_arr = np.asanyarray(pred_obj.dataobj, dtype=np.uint16)
        
        unique_labels, counts = np.unique(pred_arr, return_counts=True)
        label_names = {
            0: "Background",
            1: "PZ (Peripheral Zone)",
            2: "TZ (Transition Zone)",
            3: "CZ (Central Zone)",
            4: "AFS (Anterior Fibromuscular Stroma)",
            5: "UR (Urethra)",
            6: "PG (Periurethral Gland)",
            7: "SV_L (Seminal Vesicle Left)",
            8: "SV_R (Seminal Vesicle Right)",
            9: "Lesion (Cancer Focus)"
        }
        for lbl, count in zip(unique_labels, counts):
            vol_mm3 = count * np.prod(pred_obj.header.get_zooms()[:3])
            print(f"  Label {lbl} ({label_names.get(lbl, 'Unknown')}): {count} voxels ({vol_mm3/1000.0:.2f} cc)")
            
        # -----------------------------------------------------------------
        # EXPORT ANATOMICAL ZONES (Labels 1..8) to max_anatomy.nii.gz
        # -----------------------------------------------------------------
        zones_arr = np.copy(pred_arr)
        zones_arr[zones_arr == 9] = 0  # Remove lesion from anatomy atlas
        
        zones_nii = nib.Nifti1Image(zones_arr, pred_obj.affine, pred_obj.header)
        zones_path = os.path.join(out_anat_dir, "max_anatomy.nii.gz")
        nib.save(zones_nii, zones_path)
        
        zones_labels = {
            "1": "peripheral_zone",
            "2": "transition_zone",
            "3": "central_zone",
            "4": "anterior_fibromuscular_stroma",
            "5": "urethra",
            "6": "periurethral_gland",
            "7": "seminal_vesicle_left",
            "8": "seminal_vesicle_right"
        }
        labels_path = os.path.join(out_anat_dir, "max_anatomy_labels.json")
        with open(labels_path, "w") as f:
            json.dump(zones_labels, f, indent=4)
        print(f"  ✅ Saved anatomical zones to {zones_path}")
        print(f"  ✅ Saved zones dictionary to {labels_path}")
        
        # -----------------------------------------------------------------
        # EXPORT LESION (Label 9)
        # -----------------------------------------------------------------
        dl_lesion_arr = (pred_arr == 9).astype(np.uint16)
        dl_lesion_path = os.path.join(out_anat_dir, "prostate_dl_lesion.nii.gz")
        nib.save(nib.Nifti1Image(dl_lesion_arr, pred_obj.affine, pred_obj.header), dl_lesion_path)
        print(f"  ✅ Saved deep learning lesion mask to {dl_lesion_path}")
        
        # -----------------------------------------------------------------
        # UPDATE PET_Lesions_3.nii.gz in psma_patient_all_tp
        # Multi-reader composite:
        # 1: P01 Prostata PZ Basis links (MINT Ground Truth)
        # 2: VOI2 Lesion Focus (AI RTSS Ground Truth)
        # 3: Deep Learning Lesion Focus (nnU-Net ProstateSegmentation)
        # 4: Prostate Gland Outer Volume (Zones 1-6 Combined)
        # -----------------------------------------------------------------
        print("\n[5/5] Building composite multi-reader lesion mask for Timepoint 3...")
        orig_lesions_path = os.path.join(p3_proc, "segmentations", "t2_tra_tse_0001", "multilabel_mask.nii.gz")
        orig_obj = nib.load(orig_lesions_path)
        orig_arr = np.asanyarray(orig_obj.dataobj)
        
        # Original: 1=Gland(AI), 2=VOI2(AI), 3=Prostatavolumen(mint), 4=P01(mint), 5=Prostate(mdprostate)
        # Extract P01 (mint lesion) -> Label 1
        # Extract VOI2 (ai lesion) -> Label 2
        # Add DL Lesion -> Label 3
        # Add Prostate Gland (zones 1-6) -> Label 4
        
        composite_arr = np.zeros(pred_arr.shape, dtype=np.int16)
        
        # 1. P01 Prostata PZ Basis links (label 4 in orig)
        composite_arr[orig_arr == 4] = 1
        
        # 2. VOI2 Lesion Focus (label 2 in orig)
        composite_arr[orig_arr == 2] = 2
        
        # 3. nnU-Net DL predicted lesion
        composite_arr[pred_arr == 9] = 3
        
        # 4. Whole prostate gland envelope (zones 1..6)
        # Only where not overwritten by lesions
        gland_mask = np.isin(pred_arr, [1, 2, 3, 4, 5, 6])
        composite_arr[gland_mask & (composite_arr == 0)] = 4
        
        final_lesions_path = os.path.join(case_dir, "PET_Lesions_3.nii.gz")
        nib.save(nib.Nifti1Image(composite_arr, pred_obj.affine, pred_obj.header), final_lesions_path)
        print(f"  ✅ Saved updated PET_Lesions_3.nii.gz ({np.unique(composite_arr)}) to {final_lesions_path}")
        
    print("\n[SUCCESS] ProstateSegmentation pipeline completed successfully!")

if __name__ == "__main__":
    main()
