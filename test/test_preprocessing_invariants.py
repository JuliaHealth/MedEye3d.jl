#!/usr/bin/env python3
"""
Unit tests for Prostata_bimodal preprocessing invariants:
  1. `cads_anatomy_wiring` organ exclusion, lung-only LN override, and LN label matching.
  2. `batch_process_cohort_dual_gpu` Philips _FH (sagittal) / __701 (coronal) / _RL (axial) T2 ranking.
  3. `suvs` clinical short-lived Ga-68 (> 2 * half_life, <= 4h) acceptance vs corrupted-time rejection.
  4. `populate_radiological_reports` Excel EnglishDescription loading & metadata structure.
"""
import datetime
import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace

import h5py
import numpy as np
import SimpleITK as sitk

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "preprocessing"))

import suvs
from cads_anatomy_wiring import (
    is_excluded_non_organ_mask,
    is_lung_label_name,
    is_lymph_label_name,
    iter_authoritative_lymph_node_masks,
    remove_edge_slice_lesions,
)
from apply_edge_slices_and_semantic_filter import (
    organ_matches_mentioned,
)
from batch_process_cohort_dual_gpu import (
    classify_mri_modality,
    select_ranked_mri_sequences,
    unpack_stacked_mri_volume,
)
from populate_radiological_reports import (
    EXCEL_PATH,
    build_metadata_dict,
    load_excel_reports,
)


def _mock_pet_ds(acq_time: str, inj_dt: str, half_life: float = 4062.6, decay_corr: str = "START"):
    rph = SimpleNamespace(
        Radiopharmaceutical="PSMA",
        RadiopharmaceuticalStartTime=inj_dt[8:] + ".000000",
        RadiopharmaceuticalStartDateTime=inj_dt + ".000000",
        RadionuclideTotalDose=150_000_000.0,
        RadionuclideHalfLife=half_life,
    )
    return SimpleNamespace(
        SOPClassUID="1.2.840.10008.5.1.4.1.1.128",
        Manufacturer="SIEMENS",
        SeriesDate=inj_dt[:8],
        SeriesTime=acq_time + ".000000",
        AcquisitionDate=inj_dt[:8],
        AcquisitionTime=acq_time + ".000000",
        DecayCorrection=decay_corr,
        Units="BQML",
        PatientWeight=85.0,
        RadiopharmaceuticalInformationSequence=[rph],
    )


class TestPreprocessingInvariants(unittest.TestCase):
    def test_excluded_and_protected_anatomy_rules(self):
        for bad in ("torso_fat", "oblique_hyoid_plane", "temp", "temp_mask", "liver_temp", "helper_1"):
            self.assertTrue(is_excluded_non_organ_mask(bad), f"{bad} should be excluded")
        for good in ("temporalis_left", "temporalis_right", "temporal_lobe", "prostate", "seminal_vesicle"):
            self.assertFalse(is_excluded_non_organ_mask(good), f"{good} should NOT be excluded")

        self.assertTrue(is_lung_label_name("lung_upper_lobe_left"))
        self.assertTrue(is_lung_label_name("lung_lower_lobe_right"))
        self.assertFalse(is_lung_label_name("tracheobronchial_airways_wall"))
        self.assertFalse(is_lung_label_name("torso_fat"))

        self.assertTrue(is_lymph_label_name("inguinal_superficial_left"))
        self.assertTrue(is_lymph_label_name("thoracic_para_aortic_ln"))
        self.assertTrue(is_lymph_label_name("pelvic_lymph_node_left"))
        self.assertFalse(is_lymph_label_name("inguinal_ligament_left"))
        self.assertFalse(is_lymph_label_name("thoracic_aorta"))
        self.assertFalse(is_lymph_label_name("thoracic_duct"))

    def test_philips_t2_axial_selection(self):
        self.assertEqual(classify_mri_modality("T2_TSE_FH_Registered_to_CT.nii.gz"), ("T2", "sag"))
        self.assertEqual(classify_mri_modality("T2_TSE_RL__701_Registered_to_CT.nii.gz"), ("T2", "cor"))
        self.assertEqual(classify_mri_modality("T2_TSE_RL__901_Registered_to_CT.nii.gz"), ("T2", "cor"))
        self.assertEqual(classify_mri_modality("T2_TSE_RL_Registered_to_CT.nii.gz"), ("T2", "ax"))

        files_map = {
            "T2_TSE_FH_Registered_to_CT.nii.gz": "/tmp/fh.nii.gz",
            "T2_TSE_RL__701_Registered_to_CT.nii.gz": "/tmp/cor.nii.gz",
            "T2_TSE_RL_Registered_to_CT.nii.gz": "/tmp/rl.nii.gz",
            "T1W_TSE__Registered_to_CT.nii.gz": "/tmp/t1.nii.gz",
        }
        ranked = select_ranked_mri_sequences(files_map)
        self.assertEqual(ranked[0]["fn"], "T2_TSE_RL_Registered_to_CT.nii.gz")
        self.assertEqual(ranked[0]["plane"], "ax")

    def test_ga68_clinical_delayed_uptake_window(self):
        # 144.8 min post Ga-68 injection (like Pat80 / Pat110 / Pat128): valid!
        ds_ok = _mock_pet_ds("105245", "20230609082800", half_life=4062.6)
        inj_t, acq_t = suvs.resolve_injection_and_acquisition_times(ds_ok, 4062.6, 0.693147 / 4062.6)
        self.assertAlmostEqual((acq_t - inj_t).total_seconds(), 8685.0, places=1)

        # Corrupted > 4h gap on short-lived isotope still rejected
        ds_bad = _mock_pet_ds("135000", "20230609082800", half_life=4062.6)
        with self.assertRaises(suvs.DataStructureError):
            suvs.resolve_injection_and_acquisition_times(ds_bad, 4062.6, 0.693147 / 4062.6)

    def test_excel_radiological_dictations_coverage(self):
        recs = load_excel_reports(EXCEL_PATH)
        for pid in (1, 17, 51, 80, 87, 89, 93, 94, 95, 110, 117, 128):
            self.assertIn(pid, recs)
            md = build_metadata_dict({}, f"Pat{pid}", recs[pid], {})
            descs = [v.get("EnglishDescription", "") for v in md.values() if isinstance(v, dict)]
            self.assertTrue(any(len(str(d).strip()) > 100 for d in descs), f"Pat{pid} missing EnglishDescription")

    def test_remove_edge_slice_lesions_unconditional(self):
        import numpy as np

        arr = np.zeros((10, 8, 8), dtype=np.int16)
        # Lesion 1: purely in first 2 Z slices -> completely removed
        arr[0, 2:4, 2:4] = 1
        arr[1, 2:4, 2:4] = 1
        # Lesion 2: purely in last 2 Z slices -> completely removed
        arr[8, 2:4, 2:4] = 2
        arr[9, 2:4, 2:4] = 2
        # Lesion 3: interior + touching z=1 -> edge slices zeroed, interior preserved
        arr[1, 5, 5] = 3
        arr[4:6, 5, 5] = 3
        # Lesion 4: purely interior -> completely preserved
        arr[5, 1, 1] = 4

        removed = remove_edge_slice_lesions(arr, edge_slices=2)
        self.assertEqual(removed, [1, 2])
        self.assertEqual(int((arr[:2] > 0).sum()), 0)
        self.assertEqual(int((arr[-2:] > 0).sum()), 0)
        self.assertEqual(int((arr == 3).sum()), 2)
        self.assertEqual(int((arr == 4).sum()), 1)

    def test_semantic_organ_matching(self):
        mentioned = ["prostate", "nvseg_prostate", "iliac_left", "thyroid_gland", "liver_segment_6", "hip_left"]
        self.assertTrue(organ_matches_mentioned("Prostate", mentioned))
        self.assertTrue(organ_matches_mentioned("nvseg_prostate", mentioned))
        self.assertTrue(organ_matches_mentioned("Prostatic Urethra", mentioned))
        self.assertTrue(organ_matches_mentioned("Left Iliac Lymph Node", mentioned))
        self.assertTrue(organ_matches_mentioned("Thyroid Gland", mentioned))
        self.assertTrue(organ_matches_mentioned("nvseg_liver", mentioned))
        self.assertTrue(organ_matches_mentioned("pelvis_bone_left", mentioned))
        self.assertFalse(organ_matches_mentioned("nvseg_right_hip", mentioned))
        self.assertFalse(organ_matches_mentioned("Urinary Bladder", mentioned))
        self.assertFalse(organ_matches_mentioned("Unknown", mentioned))

    def test_authoritative_lymph_nodes_directory_precedence(self):
        with tempfile.TemporaryDirectory() as td:
            raw_dir = os.path.join(td, "raw")
            ln_dir = os.path.join(td, "lymph_nodes")
            os.makedirs(raw_dir)
            os.makedirs(ln_dir)
            shape = (4, 6, 8)  # (Z, Y, X)

            # Fallback HDF5 has an old/different mask
            old_h5 = os.path.join(raw_dir, "final_results_gpu.h5")
            with h5py.File(old_h5, "w") as hf:
                arr = np.zeros(shape, dtype=np.uint8)
                arr[0, 0, 0] = 1
                hf.create_dataset("Abdominal_Presacral", data=arr)
                hf.create_dataset("Abdominal_Perigastric", data=arr)

            # Fallback works when lymph_nodes/ absent -> switch to test fallback first via nonexistent sibling
            raw_only = os.path.join(td, "case_no_ln", "raw")
            os.makedirs(raw_only)
            with h5py.File(os.path.join(raw_only, "final_results_gpu.h5"), "w") as hf:
                arr = np.zeros(shape, dtype=np.uint8)
                arr[1, 1, 1] = 1
                hf.create_dataset("Abdominal_Presacral", data=arr)
            fb = list(iter_authoritative_lymph_node_masks(shape, raw_only))
            self.assertEqual([(n, int(m.sum()), s) for n, m, s in fb], [("abdominal_presacral", 1, "final_results_gpu.h5")])

            # Authoritative lymph_nodes/ overrides final_results_gpu.h5 (uses all_lymph_nodes + json + extra .nii.gz)
            comb = np.zeros(shape, dtype=np.uint16)
            comb[2, 2:4, 2:4] = 1  # 4 voxels
            sitk.WriteImage(sitk.GetImageFromArray(comb), os.path.join(ln_dir, "all_lymph_nodes.nii.gz"))
            with open(os.path.join(ln_dir, "lymph_node_labels.json"), "w") as jf:
                json.dump({"1": "Abdominal_Presacral"}, jf)
            extra = np.zeros(shape, dtype=np.uint8)
            extra[3, 1, 1:3] = 1  # 2 voxels
            sitk.WriteImage(sitk.GetImageFromArray(extra), os.path.join(ln_dir, "Abdominal_Obturator_Left.nii.gz"))

            auth = {n: (int(m.sum()), s) for n, m, s in iter_authoritative_lymph_node_masks(shape, raw_dir)}
            self.assertEqual(auth["abdominal_presacral"], (4, "lymph_nodes/all_lymph_nodes.nii.gz"))
            self.assertEqual(auth["abdominal_obturator_left"], (2, "lymph_nodes/Abdominal_Obturator_Left.nii.gz"))
            self.assertNotIn("abdominal_perigastric", auth)

    def test_adc_truncated_names_and_interleaved_dwi_unpacking(self):
        # 1. Truncated Slicer ADC names & Philips/Siemens dDWI/IsoADC classify as ADC
        for fn in (
            "s)_Registered_to_CT.nii.gz",
            "s):Jan_23_2023_12-37-51_CET_Registered_to_CT.nii.gz",
            "dDWI_SENSE.nii.gz",
            "IsoADC_Registered_to_CT.nii.gz",
        ):
            self.assertEqual(classify_mri_modality(fn), ("ADC", "ax"), fn)
        self.assertEqual(classify_mri_modality("DWI_tra_Registered_to_CT.nii.gz"), ("DWI", "ax"))

        # 2. Slice-interleaved DWI (even=b0 bright ~250, odd=b800 dark ~80) extracts b800 & doubles sp_z
        arr = np.zeros((44, 6, 6), dtype=np.float32)  # (Z, Y, X)
        arr[0::2] = 250.0
        arr[1::2] = 80.0
        im = sitk.GetImageFromArray(arr)
        im.SetSpacing((1.5, 1.5, 2.25))
        im.SetOrigin((10.0, 20.0, 30.0))

        unpacked = unpack_stacked_mri_volume(im, "DWI", "DWI_tra_Registered_to_CT.nii.gz")
        self.assertEqual(unpacked.GetSize(), (6, 6, 22))
        self.assertAlmostEqual(unpacked.GetSpacing()[2], 4.5, places=3)
        u_arr = sitk.GetArrayFromImage(unpacked)
        self.assertAlmostEqual(float(u_arr.mean()), 80.0, places=3)

    def test_eadc_deprioritization_rgb_and_dataset501_zone_map(self):
        # 1. Quantitative ADC must outrank exponential eADC; RGB Philips WIP perfusion colormaps excluded; dwip -> ADC
        self.assertEqual(classify_mri_modality("dWIP_DWI_UKE_Zoom.nii.gz"), ("ADC", "ax"))
        for wip in (
            "pAREAWIP_DCE_20_SENSE_Registered_to_CT.nii.gz",
            "pREWIP_DCE_20_SENSE_Registered_to_CT.nii.gz",
            "pTTPWIP_DCE_20_SENSE_Registered_to_CT.nii.gz",
        ):
            self.assertIsNone(classify_mri_modality(wip))
        ranked = select_ranked_mri_sequences({
            "T2_tra_Registered_to_CT.nii.gz": "/tmp/t2.nii.gz",
            "eADC:Jun_15_Registered_to_CT.nii.gz": "/tmp/eadc.nii.gz",
            "ADC_Recovered_Registered_to_CT.nii.gz": "/tmp/adc.nii.gz",
        })
        adc_pick = next(s for s in ranked if s["mod"] == "ADC")
        self.assertEqual(adc_pick["fn"], "ADC_Recovered_Registered_to_CT.nii.gz")

        # 2. Native MRI lesion source is strictly Dataset501 Label 9 (no NNUNET_OR_Processed_Lesions_MR fallback)
        from run_prostate_seg_missing import PROSTATE_ZONE_LABELS
        self.assertEqual(sorted(PROSTATE_ZONE_LABELS.keys()), [str(i) for i in range(1, 9)])
        with open(os.path.join(REPO_ROOT, "scripts", "preprocessing", "batch_process_cohort_dual_gpu.py")) as bf:
            batch_src = bf.read()
        self.assertNotIn('mr_lesions_dest = os.path.join(raw_dir, "NNUNET_OR_Processed_Lesions_MR.seg.nrrd")', batch_src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
