#!/usr/bin/env python3
"""
prepare_psma_case_for_medeye3d.py

Assembles the multi-patient PSMA Köln dataset into a single longitudinal multi-timepoint
case folder at data/cases/psma_patient_all_tp/:
- TP 0: Patient 2 Early Whole-Body PET/CT (12 lesions)
- TP 1: Patient 2 Late Whole-Body PET/CT (11 lesions)
- TP 2: Patient 1 Late Whole-Body PET/CT
- TP 3: Patient 3 Prostate mpMRI (T2 Tra + Zoomit ADC + 5 multi-reader segments)

Preserves real clinical lesion names in scene_hierarchy.json and builds
metadata.json and matches.json.
"""

import os
import sys
import json
import shutil
import numpy as np
import nibabel as nib


def make_identity_tfm(filepath):
    """Write an ITK identity Affine transform file."""
    content = """#Insight Transform File V1.0
#Transform 0
Transform: AffineTransform_double_3_3
Parameters: 1 0 0 0 1 0 0 0 1 0 0 0
FixedParameters: 0 0 0
"""
    with open(filepath, "w") as f:
        f.write(content)


def main():
    base_data = "/mnt/big/project_ssd/project_ssd/MedEye3d.jl/data"
    proc_dir = os.path.join(base_data, "processed")
    cases_dir = os.path.join(base_data, "cases")
    out_dir = os.path.join(cases_dir, "psma_patient_all_tp")
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"[Step 1] Preparing multi-timepoint case at {out_dir}...")
    
    # -------------------------------------------------------------
    # TIMEPOINT 0: Patient 2 Early Whole-Body PET/CT (Baseline)
    # -------------------------------------------------------------
    p2_dir = os.path.join(proc_dir, "Patient_2_277735")
    p2_ct = os.path.join(p2_dir, "CT", "ct_ac_wb_3.0_hd_fov.nii.gz")
    p2_pet = os.path.join(p2_dir, "PET", "pet_wb.nii.gz")
    p2_seg = os.path.join(p2_dir, "PET", "segmentations", "pet_wb", "multilabel_mask.nii.gz")
    p2_json = os.path.join(p2_dir, "PET", "segmentations", "pet_wb", "labels_index.json")
    
    shutil.copyfile(p2_ct, os.path.join(out_dir, "Fixed_CT_Volume_0.nii.gz"))
    shutil.copyfile(p2_pet, os.path.join(out_dir, "SUV_PET_Image_0.nii.gz"))
    shutil.copyfile(p2_seg, os.path.join(out_dir, "PET_Lesions_0.nii.gz"))
    
    with open(p2_json, "r") as f:
        meta0 = json.load(f)
    tp0_segments = [roi["roi_name"] for roi in meta0["rois"]]
    print(f"  TP 0: Copied CT, PET, and {len(tp0_segments)} lesions.")
    
    # -------------------------------------------------------------
    # TIMEPOINT 1: Patient 2 Late Whole-Body PET/CT (Follow-up 1)
    # -------------------------------------------------------------
    p2_ct_late = os.path.join(p2_dir, "CT", "ct_ac_wb_spaet.nii.gz")
    p2_pet_late = os.path.join(p2_dir, "PET", "pet_wb_spaet.nii.gz")
    p2_seg_late = os.path.join(p2_dir, "PET", "segmentations", "pet_wb_spaet", "multilabel_mask.nii.gz")
    p2_json_late = os.path.join(p2_dir, "PET", "segmentations", "pet_wb_spaet", "labels_index.json")
    
    shutil.copyfile(p2_ct_late, os.path.join(out_dir, "Fixed_CT_Volume_1.nii.gz"))
    shutil.copyfile(p2_pet_late, os.path.join(out_dir, "SUV_PET_Image_1.nii.gz"))
    shutil.copyfile(p2_seg_late, os.path.join(out_dir, "PET_Lesions_1.nii.gz"))
    make_identity_tfm(os.path.join(out_dir, "Transform_FollowUp_to_Baseline_1.tfm"))
    
    with open(p2_json_late, "r") as f:
        meta1 = json.load(f)
    tp1_segments = [roi["roi_name"] for roi in meta1["rois"]]
    print(f"  TP 1: Copied CT, PET, and {len(tp1_segments)} lesions.")
    
    # -------------------------------------------------------------
    # TIMEPOINT 2: Patient 1 Late Whole-Body PET/CT (Follow-up 2)
    # -------------------------------------------------------------
    p1_dir = os.path.join(proc_dir, "Patient_1_277820")
    p1_ct = os.path.join(p1_dir, "CT", "ct_ac_wb_spaet.nii.gz")
    p1_pet = os.path.join(p1_dir, "PET", "pet_wb_spaet.nii.gz")
    
    shutil.copyfile(p1_ct, os.path.join(out_dir, "Fixed_CT_Volume_2.nii.gz"))
    shutil.copyfile(p1_pet, os.path.join(out_dir, "SUV_PET_Image_2.nii.gz"))
    
    # Create empty mask for TP 2 matching PET
    p1_pet_obj = nib.load(p1_pet)
    empty_mask = np.zeros(p1_pet_obj.shape, dtype=np.int16)
    nib.save(nib.Nifti1Image(empty_mask, p1_pet_obj.affine, p1_pet_obj.header), os.path.join(out_dir, "PET_Lesions_2.nii.gz"))
    make_identity_tfm(os.path.join(out_dir, "Transform_FollowUp_to_Baseline_2.tfm"))
    tp2_segments = []
    print(f"  TP 2: Copied CT, PET.")
    
    # -------------------------------------------------------------
    # TIMEPOINT 3: Patient 3 mpMRI T2W Axial (Follow-up 3)
    # -------------------------------------------------------------
    p3_dir = os.path.join(proc_dir, "Patient_3_278803")
    p3_mr_t2 = os.path.join(p3_dir, "MRI", "t2_tra_tse_0001.nii.gz")
    p3_mr_adc = os.path.join(p3_dir, "MRI", "zoomit_diff_tra_adc.nii.gz")
    p3_mr_dwi = os.path.join(p3_dir, "MRI", "zoomit_diff_tra_calc_bval.nii.gz")
    p3_mr_t1 = os.path.join(p3_dir, "MRI", "km_t1_tra_vibe_dixon_w.nii.gz")
    p3_seg = os.path.join(p3_dir, "MRI", "segmentations", "t2_tra_tse_0001", "multilabel_mask.nii.gz")
    p3_json = os.path.join(p3_dir, "MRI", "segmentations", "t2_tra_tse_0001", "labels_index.json")
    
    with open(p3_json, "r") as f:
        meta3 = json.load(f)
    tp3_segments = [f"{roi['roi_name']} ({roi['source_system'].upper()})" for roi in meta3["rois"]]

    # TP 3: T2W Axial TSE + ADC overlay
    shutil.copyfile(p3_mr_t2, os.path.join(out_dir, "Fixed_CT_Volume_3.nii.gz"))
    shutil.copyfile(p3_mr_adc, os.path.join(out_dir, "SUV_PET_Image_3.nii.gz"))
    shutil.copyfile(p3_seg, os.path.join(out_dir, "PET_Lesions_3.nii.gz"))
    if not os.path.isfile(os.path.join(out_dir, "Transform_FollowUp_to_Baseline_3.tfm")):
        make_identity_tfm(os.path.join(out_dir, "Transform_FollowUp_to_Baseline_3.tfm"))
    print(f"  TP 3: Copied MRI T2, ADC overlay, and {len(tp3_segments)} segments.")

    # TP 4: mpMRI ADC Map
    shutil.copyfile(p3_mr_adc, os.path.join(out_dir, "Fixed_CT_Volume_4.nii.gz"))
    shutil.copyfile(p3_mr_adc, os.path.join(out_dir, "SUV_PET_Image_4.nii.gz"))
    shutil.copyfile(p3_seg, os.path.join(out_dir, "PET_Lesions_4.nii.gz"))
    shutil.copyfile(os.path.join(out_dir, "Transform_FollowUp_to_Baseline_3.tfm"), os.path.join(out_dir, "Transform_FollowUp_to_Baseline_4.tfm"))
    print(f"  TP 4: Copied MRI ADC Map.")

    # TP 5: mpMRI DWI b2000
    shutil.copyfile(p3_mr_dwi, os.path.join(out_dir, "Fixed_CT_Volume_5.nii.gz"))
    shutil.copyfile(p3_mr_dwi, os.path.join(out_dir, "SUV_PET_Image_5.nii.gz"))
    shutil.copyfile(p3_seg, os.path.join(out_dir, "PET_Lesions_5.nii.gz"))
    shutil.copyfile(os.path.join(out_dir, "Transform_FollowUp_to_Baseline_3.tfm"), os.path.join(out_dir, "Transform_FollowUp_to_Baseline_5.tfm"))
    print(f"  TP 5: Copied MRI DWI b=2000.")

    # TP 6: mpMRI T1W Contrast-Enhanced
    shutil.copyfile(p3_mr_t1, os.path.join(out_dir, "Fixed_CT_Volume_6.nii.gz"))
    shutil.copyfile(p3_mr_t1, os.path.join(out_dir, "SUV_PET_Image_6.nii.gz"))
    shutil.copyfile(p3_seg, os.path.join(out_dir, "PET_Lesions_6.nii.gz"))
    shutil.copyfile(os.path.join(out_dir, "Transform_FollowUp_to_Baseline_3.tfm"), os.path.join(out_dir, "Transform_FollowUp_to_Baseline_6.tfm"))
    print(f"  TP 6: Copied MRI T1W Axial CE.")
    
    # -------------------------------------------------------------
    # BUILD scene_hierarchy.json
    # -------------------------------------------------------------
    hierarchy = [
        # Baseline TP 0
        {
            "name": "Fixed_CT_Volume_0",
            "type": "vtkMRMLScalarVolumeNode",
            "modality": "CT"
        },
        {
            "name": "SUV_PET_Image_0",
            "type": "vtkMRMLScalarVolumeNode",
            "modality": "PET"
        },
        {
            "name": "PET_Lesions_0",
            "type": "vtkMRMLSegmentationNode",
            "segments": tp0_segments
        },
        {
            "name": "max_anatomy_fixed_ct_0",
            "source": "anatomy_out_fixed_ct_0/max_anatomy.nii.gz",
            "labels": "anatomy_out_fixed_ct_0/max_anatomy_labels.json",
            "type": "vtkMRMLSegmentationNode"
        },
        {
            "name": "skellytour_fixed_ct_0",
            "source": "Skellytour_0.nii.gz",
            "type": "vtkMRMLSegmentationNode"
        },
        
        # Follow-up 1 (TP 1)
        {
            "name": "Transform_FollowUp_to_Baseline_1",
            "type": "vtkMRMLLinearTransformNode",
            "transform_type": "Rigid/Affine (Linear)",
            "children": [
                {
                    "name": "Fixed_CT_Volume_1",
                    "type": "vtkMRMLScalarVolumeNode",
                    "modality": "CT"
                },
                {
                    "name": "SUV_PET_Image_1",
                    "type": "vtkMRMLScalarVolumeNode",
                    "modality": "PET"
                },
                {
                    "name": "PET_Lesions_1",
                    "type": "vtkMRMLSegmentationNode",
                    "segments": tp1_segments
                }
            ]
        },
        
        # Follow-up 2 (TP 2)
        {
            "name": "Transform_FollowUp_to_Baseline_2",
            "type": "vtkMRMLLinearTransformNode",
            "transform_type": "Rigid/Affine (Linear)",
            "children": [
                {
                    "name": "Fixed_CT_Volume_2",
                    "type": "vtkMRMLScalarVolumeNode",
                    "modality": "CT"
                },
                {
                    "name": "SUV_PET_Image_2",
                    "type": "vtkMRMLScalarVolumeNode",
                    "modality": "PET"
                },
                {
                    "name": "PET_Lesions_2",
                    "type": "vtkMRMLSegmentationNode",
                    "segments": tp2_segments
                }
            ]
        },
        
        # Follow-up 3 (TP 3 - mpMRI T2W)
        {
            "name": "Transform_FollowUp_to_Baseline_3",
            "type": "vtkMRMLLinearTransformNode",
            "transform_type": "Rigid/Affine (Linear)",
            "modality": "T2",
            "children": [
                {
                    "name": "Fixed_CT_Volume_3",
                    "type": "vtkMRMLScalarVolumeNode",
                    "modality": "T2"
                },
                {
                    "name": "SUV_PET_Image_3",
                    "type": "vtkMRMLScalarVolumeNode",
                    "modality": "ADC"
                },
                {
                    "name": "PET_Lesions_3",
                    "type": "vtkMRMLSegmentationNode",
                    "segments": tp3_segments
                }
            ]
        },

        # Follow-up 4 (TP 4 - mpMRI ADC)
        {
            "name": "Transform_FollowUp_to_Baseline_4",
            "type": "vtkMRMLLinearTransformNode",
            "transform_type": "Rigid/Affine (Linear)",
            "modality": "ADC",
            "children": [
                {
                    "name": "Fixed_CT_Volume_4",
                    "type": "vtkMRMLScalarVolumeNode",
                    "modality": "ADC"
                },
                {
                    "name": "SUV_PET_Image_4",
                    "type": "vtkMRMLScalarVolumeNode",
                    "modality": "ADC"
                },
                {
                    "name": "PET_Lesions_4",
                    "type": "vtkMRMLSegmentationNode",
                    "segments": tp3_segments
                }
            ]
        },

        # Follow-up 5 (TP 5 - mpMRI DWI)
        {
            "name": "Transform_FollowUp_to_Baseline_5",
            "type": "vtkMRMLLinearTransformNode",
            "transform_type": "Rigid/Affine (Linear)",
            "modality": "DWI",
            "children": [
                {
                    "name": "Fixed_CT_Volume_5",
                    "type": "vtkMRMLScalarVolumeNode",
                    "modality": "DWI"
                },
                {
                    "name": "SUV_PET_Image_5",
                    "type": "vtkMRMLScalarVolumeNode",
                    "modality": "DWI"
                },
                {
                    "name": "PET_Lesions_5",
                    "type": "vtkMRMLSegmentationNode",
                    "segments": tp3_segments
                }
            ]
        },

        # Follow-up 6 (TP 6 - mpMRI T1W CE)
        {
            "name": "Transform_FollowUp_to_Baseline_6",
            "type": "vtkMRMLLinearTransformNode",
            "transform_type": "Rigid/Affine (Linear)",
            "modality": "T1",
            "children": [
                {
                    "name": "Fixed_CT_Volume_6",
                    "type": "vtkMRMLScalarVolumeNode",
                    "modality": "T1"
                },
                {
                    "name": "SUV_PET_Image_6",
                    "type": "vtkMRMLScalarVolumeNode",
                    "modality": "T1"
                },
                {
                    "name": "PET_Lesions_6",
                    "type": "vtkMRMLSegmentationNode",
                    "segments": tp3_segments
                }
            ]
        }
    ]
    
    with open(os.path.join(out_dir, "scene_hierarchy.json"), "w", encoding="utf-8") as f:
        json.dump(hierarchy, f, indent=4, ensure_ascii=False)
    print("  Created scene_hierarchy.json with 7 timepoints.")
    
    # -------------------------------------------------------------
    # BUILD metadata.json
    # -------------------------------------------------------------
    metadata = [
        {"BirthdayDate": "1966-03-07"},
        {
            "20260407": {
                "CT": {
                    "name": "Fixed_CT_Volume_0",
                    "Modality": "CT",
                    "StudyDate": "20260407",
                    "SeriesDescription": "CT AC WB 3.0 HD FoV",
                    "Manufacturer": "SIEMENS"
                },
                "PET": {
                    "name": "SUV_PET_Image_0",
                    "Modality": "PET",
                    "StudyDate": "20260407",
                    "SeriesDescription": "PET WB Early (SUV)",
                    "Manufacturer": "SIEMENS"
                }
            }
        },
        {
            "20260407_late": {
                "CT": {
                    "name": "Fixed_CT_Volume_1",
                    "Modality": "CT",
                    "StudyDate": "20260407",
                    "SeriesDescription": "CT AC WB Late",
                    "Manufacturer": "SIEMENS"
                },
                "PET": {
                    "name": "SUV_PET_Image_1",
                    "Modality": "PET",
                    "StudyDate": "20260407",
                    "SeriesDescription": "PET WB Late (SUV)",
                    "Manufacturer": "SIEMENS"
                }
            }
        },
        {
            "20260414": {
                "CT": {
                    "name": "Fixed_CT_Volume_2",
                    "Modality": "CT",
                    "StudyDate": "20260414",
                    "SeriesDescription": "CT AC WB Late",
                    "Manufacturer": "SIEMENS"
                },
                "PET": {
                    "name": "SUV_PET_Image_2",
                    "Modality": "PET",
                    "StudyDate": "20260414",
                    "SeriesDescription": "PET WB Late (SUV)",
                    "Manufacturer": "SIEMENS"
                }
            }
        },
        {
            "20260428_t2": {
                "CT": {
                    "name": "Fixed_CT_Volume_3",
                    "Modality": "T2",
                    "StudyDate": "20260428",
                    "SeriesDescription": "T2 Tra TSE (Axial High-Res)",
                    "Manufacturer": "SIEMENS"
                },
                "PET": {
                    "name": "SUV_PET_Image_3",
                    "Modality": "ADC",
                    "StudyDate": "20260428",
                    "SeriesDescription": "Zoomit Diffusion ADC High-Res",
                    "Manufacturer": "SIEMENS"
                }
            }
        },
        {
            "20260428_adc": {
                "CT": {
                    "name": "Fixed_CT_Volume_4",
                    "Modality": "ADC",
                    "StudyDate": "20260428",
                    "SeriesDescription": "Zoomit Diffusion ADC Map",
                    "Manufacturer": "SIEMENS"
                },
                "PET": {
                    "name": "SUV_PET_Image_4",
                    "Modality": "ADC",
                    "StudyDate": "20260428",
                    "SeriesDescription": "Zoomit Diffusion ADC Map",
                    "Manufacturer": "SIEMENS"
                }
            }
        },
        {
            "20260428_dwi": {
                "CT": {
                    "name": "Fixed_CT_Volume_5",
                    "Modality": "DWI",
                    "StudyDate": "20260428",
                    "SeriesDescription": "Zoomit Diffusion DWI b2000",
                    "Manufacturer": "SIEMENS"
                },
                "PET": {
                    "name": "SUV_PET_Image_5",
                    "Modality": "DWI",
                    "StudyDate": "20260428",
                    "SeriesDescription": "Zoomit Diffusion DWI b2000",
                    "Manufacturer": "SIEMENS"
                }
            }
        },
        {
            "20260428_t1": {
                "CT": {
                    "name": "Fixed_CT_Volume_6",
                    "Modality": "T1",
                    "StudyDate": "20260428",
                    "SeriesDescription": "KM T1 Tra VIBE Dixon Contrast",
                    "Manufacturer": "SIEMENS"
                },
                "PET": {
                    "name": "SUV_PET_Image_6",
                    "Modality": "T1",
                    "StudyDate": "20260428",
                    "SeriesDescription": "KM T1 Tra VIBE Dixon Contrast",
                    "Manufacturer": "SIEMENS"
                }
            }
        }
    ]
    
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)
    print("  Created metadata.json.")
    
    # -------------------------------------------------------------
    # BUILD matches.json
    # -------------------------------------------------------------
    matches = [
        {
            "name": "PET_Lesions_0",
            "raw_lesion": "Segment_8",
            "node": "parent",
            "lesion": "PT01 ProstataCa Gl 7a gesichert",
            "volume_mm3": 17469.0,
            "children": [
                {
                    "name": "PET_Lesions_1",
                    "raw_lesion": "Segment_1",
                    "lesion": "PT02 ProstataCa Gl 3+4=7a spaet",
                    "volume_mm3": 7960.0,
                    "match_type": "PROXIMITY",
                    "group_id": 1
                },
                {
                    "name": "PET_Lesions_3",
                    "raw_lesion": "Segment_4",
                    "lesion": "P01 Prostata PZ Basis links (MINT)",
                    "volume_mm3": 340.0,
                    "match_type": "PROXIMITY",
                    "group_id": 1
                }
            ]
        },
        {
            "name": "PET_Lesions_0",
            "raw_lesion": "Segment_1",
            "node": "parent",
            "lesion": "F08 Knochen Beckenguertel links UBU",
            "volume_mm3": 28270.0,
            "children": [
                {
                    "name": "PET_Lesions_1",
                    "raw_lesion": "Segment_4",
                    "lesion": "F12 Knochen Beckenguertel links UBU spaet",
                    "volume_mm3": 500.0,
                    "match_type": "PROXIMITY",
                    "group_id": 2
                }
            ]
        },
        {
            "name": "PET_Lesions_0",
            "raw_lesion": "Segment_5",
            "node": "parent",
            "lesion": "LN04 Lymphknoten iliaca externa links",
            "volume_mm3": 747.0,
            "children": [
                {
                    "name": "PET_Lesions_1",
                    "raw_lesion": "Segment_5",
                    "lesion": "LN07 Lymphknoten iliaca externa links",
                    "volume_mm3": 1140.0,
                    "match_type": "PROXIMITY",
                    "group_id": 3
                }
            ]
        }
    ]
    
    with open(os.path.join(out_dir, "matches.json"), "w", encoding="utf-8") as f:
        json.dump(matches, f, indent=4, ensure_ascii=False)
    print("  Created matches.json.")
    print(f"\n[Done] Case setup complete at {out_dir}")


if __name__ == "__main__":
    main()
