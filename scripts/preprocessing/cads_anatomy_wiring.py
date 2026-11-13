#!/usr/bin/env python3
"""
cads_anatomy_wiring.py - Integrates CADS Task 556 (RT Risk Organs) and Task 558 (Head & Neck)
structures into the MedEye3d max_anatomy volume and metadata.

Target structures:
  - Task 556 (GC256):
      * Index 1: spinal_canal       -> Label 538
      * Index 5: sigmoid            -> Label 532 (sigmoid_colon, carves out of colon)
      * Index 8: seminal_vesicle    -> Label 531
  - Task 558 (OAR258):
      * Index 16: OAR_Glnd_Lacrimal_L -> Label 533 (lacrimal_gland_left)
      * Index 17: OAR_Glnd_Lacrimal_R -> Label 534 (lacrimal_gland_right)
      * Index 24: OAR_OpticChiasm    -> Label 535 (optic_chiasm)
      * Index 25: OAR_OpticNrv_L     -> Label 536 (optic_nerve_left)
      * Index 26: OAR_OpticNrv_R     -> Label 537 (optic_nerve_right)
      * Index 29: OAR_Pituitary      -> Label 539 (pituitary_gland)
"""

import os
import sys
import json
import time
import h5py
import numpy as np
import SimpleITK as sitk

# Label ID definitions to be merged into max_anatomy_labels.json
CADS_STRUCTURE_MAP = {
    531: "seminal_vesicle",
    532: "sigmoid_colon",
    533: "lacrimal_gland_left",
    534: "lacrimal_gland_right",
    535: "optic_chiasm",
    536: "optic_nerve_left",
    537: "optic_nerve_right",
    538: "spinal_canal",
    539: "pituitary_gland",
}

# Task 556 source indices -> new max_anatomy label IDs
TASK_556_MAPPING = {
    1: 538,  # spinal_canal
    5: 532,  # sigmoid -> sigmoid_colon
    8: 531,  # seminal_vesicle
}

# Task 558 source indices -> new max_anatomy label IDs
TASK_558_MAPPING = {
    16: 533,  # OAR_Glnd_Lacrimal_L
    17: 534,  # OAR_Glnd_Lacrimal_R
    24: 535,  # OAR_OpticChiasm
    25: 536,  # OAR_OpticNrv_L
    26: 537,  # OAR_OpticNrv_R
    29: 539,  # OAR_Pituitary
}


def resample_mask_to_reference(mask_img: sitk.Image, ref_img: sitk.Image) -> np.ndarray:
    """Resamples a segmentation mask onto ref_img geometry using nearest neighbor interpolation."""
    if mask_img.GetSize() != ref_img.GetSize() or mask_img.GetSpacing() != ref_img.GetSpacing() or mask_img.GetOrigin() != ref_img.GetOrigin():
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(ref_img)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetDefaultPixelValue(0)
        resampler.SetOutputPixelType(sitk.sitkUInt16)
        mask_resampled = resampler.Execute(mask_img)
    else:
        mask_resampled = mask_img
    return sitk.GetArrayFromImage(mask_resampled)


SIGMOID_LABEL = 532

_LYMPH_PREFIXES = ("neck_", "thoracic_", "axillary_", "abdominal_",
                   "deep_inguinal", "superficial_inguinal", "inguinal_")


def is_lymph_label_name(name: str) -> bool:
    """Name-based lymph node area test (mirrors LesionAssociation.is_lymph_node_structure).

    Excludes non-lymph helper/structural labels that share station prefixes
    (e.g. 'abdominal_cavity', 'thoracic_cavity', 'thoracic_chest_wall',
    'thoracic_sternum_exclusion_zone', '*_helper', '*_bridge').
    """
    ln = (name or "").lower()
    if not ln:
        return False
    if any(bad in ln for bad in ("_cavity", "exclusion_zone", "chest_wall", "_helper", "_bridge", "_base", "_dilated")):
        return False
    if "lymph" in ln or "knoten" in ln or "node" in ln:
        return True
    return ln.startswith(_LYMPH_PREFIXES)


def lymph_label_ids(labels_dict: dict) -> list:
    return [int(k) for k, v in labels_dict.items() if is_lymph_label_name(v)]


def colon_label_ids(labels_dict: dict) -> list:
    """Plain colon labels ('colon', 'nvseg_colon'); excludes the sigmoid itself."""
    return [int(k) for k, v in labels_dict.items()
            if "colon" in v.lower() and "sigmoid" not in v.lower()]


def resolve_cads_label_ids(labels_dict: dict) -> dict:
    """Map default CADS ids (531..539) -> ids that are actually free in this case.

    Some cases (e.g. Pat1, Pat6, Pat19, Pat20) already use ids >= 531 for real
    lymph areas, so the fixed defaults would collide. Resolution per structure:
      1. already registered under its name  -> keep that id (idempotent re-runs)
      2. default id unused                  -> use the default
      3. default id taken by another label  -> allocate max(existing ids)+1, ...
    """
    by_name = {v: int(k) for k, v in labels_dict.items()}
    used = {int(k) for k in labels_dict.keys()}
    next_free = max(used | {539}) + 1
    out = {}
    for default_id, name in CADS_STRUCTURE_MAP.items():
        if name in by_name:
            out[default_id] = by_name[name]
        elif default_id not in used:
            out[default_id] = default_id
            used.add(default_id)
        else:
            while next_free in used:
                next_free += 1
            out[default_id] = next_free
            used.add(next_free)
    return out


def merge_cads_structures(base_anatomy_arr: np.ndarray,
                          cads_556_arr: np.ndarray = None,
                          cads_558_arr: np.ndarray = None,
                          labels_dict: dict = None) -> tuple:
    """
    Merges CADS Task 556 and Task 558 segmentations into max_anatomy array.

    Rules:
      - Lymph node areas are PROTECTED: no CADS structure may overwrite a voxel that
        is a lymph node area in the base anatomy (they stay fully visible).
      - Sigmoid colon replaces only the overlapping part of the plain colon
        (labels 'colon' / 'nvseg_colon') and otherwise-empty voxels; the rest of
        the colon is kept as colon. It never replaces other organs.
      - Every other CADS structure overwrites non-lymph voxels only.
      - CADS label ids never collide with existing labels (see resolve_cads_label_ids).

    Returns:
      (merged_anatomy_arr, updated_labels_dict)
    """
    out_arr = base_anatomy_arr.copy()
    if labels_dict is None:
        labels_dict = {}
    else:
        labels_dict = dict(labels_dict)

    # Lymph / colon ids are resolved from the BASE labels (before CADS names are added)
    ln_ids = lymph_label_ids(labels_dict)
    colon_ids = colon_label_ids(labels_dict)
    protected = np.isin(base_anatomy_arr, ln_ids) if ln_ids else np.zeros(base_anatomy_arr.shape, dtype=bool)
    colon_or_empty = np.isin(base_anatomy_arr, colon_ids) | (base_anatomy_arr == 0)

    # Collision-free ids, then register all CADS structure names
    id_map = resolve_cads_label_ids(labels_dict)
    for default_id, name in CADS_STRUCTURE_MAP.items():
        labels_dict[str(id_map[default_id])] = name

    def _apply(mask_voxels, default_label):
        allowed = mask_voxels & ~protected
        if default_label == SIGMOID_LABEL:
            allowed &= colon_or_empty
        if np.any(allowed):
            out_arr[allowed] = id_map[default_label]

    # 1. Merge Task 556 (RT Risk Organs: spinal_canal, sigmoid, seminal_vesicle)
    if cads_556_arr is not None:
        for src_idx, target_label in TASK_556_MAPPING.items():
            _apply(cads_556_arr == src_idx, target_label)

    # 2. Merge Task 558 (Head & Neck: lacrimal glands, optic chiasm/nerves, pituitary)
    if cads_558_arr is not None:
        for src_idx, target_label in TASK_558_MAPPING.items():
            _apply(cads_558_arr == src_idx, target_label)

    return out_arr, labels_dict


def run_cads_inference_for_ct(ct_path: str,
                              output_dir: str,
                              gpu_id: int = 0,
                              license_type: str = "research") -> dict:
    """
    Executes CADS inference for Task 556 and Task 558 on a single CT volume.
    Returns dictionary with paths to resulting NIfTI files:
      {'task_556': path, 'task_558': path}
    """
    os.makedirs(output_dir, exist_ok=True)
    cads_root = "/workspaces/MedEye3d.jl/third_party/CADS"
    if cads_root not in sys.path:
        sys.path.insert(0, cads_root)

    weights_dir = "/root/.cads/model_weights"
    os.environ["CADS_WEIGHTS_PATH"] = weights_dir
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    from cads.utils.libs import setup_nnunet_env, check_or_download_model_weights
    setup_nnunet_env(license_type)

    results = {}
    for task_id in [556, 558]:
        check_or_download_model_weights(task_id, license=license_type)
        task_out = os.path.join(output_dir, f"cads_part_{task_id}.nii.gz")
        # Run predict script
        cmd = f"python3 {cads_root}/cads/scripts/predict_images.py -in {ct_path} -out {output_dir} -task {task_id} -license {license_type} --no-postprocessing"
        print(f"[CADS] Running Task {task_id} inference: {cmd}")
        ret = os.system(cmd)
        if ret == 0 and os.path.exists(task_out):
            results[f"task_{task_id}"] = task_out
            print(f"[CADS] Task {task_id} completed -> {task_out}")
        else:
            print(f"[CADS] Warning: Task {task_id} exited with status {ret}")

    return results


def patch_case_h5_with_cads(h5_path: str,
                            cads_556_path: str = None,
                            cads_558_path: str = None,
                            ref_ct_path: str = None,
                            base_anat_override: np.ndarray = None,
                            base_labels_override: dict = None) -> bool:
    """
    Injects CADS Task 556 and Task 558 segmentations into an existing preprocessed_volumes.h5.
    Handles transverse axis-1 pre-flipping to match the MedEye3d coordinate system.
    """
    if not os.path.exists(h5_path):
        print(f"[ERROR] HDF5 file not found: {h5_path}")
        return False

    ref_img = None
    if ref_ct_path and os.path.exists(ref_ct_path):
        ref_img = sitk.ReadImage(ref_ct_path)

    cads_556_arr = None
    if cads_556_path and os.path.exists(cads_556_path):
        img_556 = sitk.ReadImage(cads_556_path)
        if ref_img is not None:
            cads_556_arr = resample_mask_to_reference(img_556, ref_img)
        else:
            cads_556_arr = sitk.GetArrayFromImage(img_556)
        # Apply MedEye3d transverse axis-1 pre-flip (flip Y axis)
        cads_556_arr = np.flip(cads_556_arr, axis=1)

    cads_558_arr = None
    if cads_558_path and os.path.exists(cads_558_path):
        img_558 = sitk.ReadImage(cads_558_path)
        if ref_img is not None:
            cads_558_arr = resample_mask_to_reference(img_558, ref_img)
        else:
            cads_558_arr = sitk.GetArrayFromImage(img_558)
        # Apply MedEye3d transverse axis-1 pre-flip (flip Y axis)
        cads_558_arr = np.flip(cads_558_arr, axis=1)

    with h5py.File(h5_path, "r+") as f:
        # 1. Read existing max_anatomy (or the clean rebuilt base when repairing)
        if base_anat_override is not None:
            base_anat = base_anat_override
        elif "ATLAS/max_anatomy" in f:
            base_anat = f["ATLAS/max_anatomy"][:]
        elif "BASELINE/max_anatomy.nii.gz" in f:
            base_anat = f["BASELINE/max_anatomy.nii.gz"][:]
        else:
            print(f"[ERROR] No max_anatomy found in {h5_path}")
            return False

        # 2. Read existing max_anatomy labels
        labels_dict = {}
        if base_labels_override is not None:
            labels_dict = dict(base_labels_override)
        elif "_meta_/max_anatomy_labels.json" in f:
            try:
                labels_dict = json.loads(f["_meta_/max_anatomy_labels.json"][()])
            except Exception:
                pass

        # 3. Merge CADS structures
        merged_anat, updated_labels = merge_cads_structures(
            base_anat,
            cads_556_arr=cads_556_arr,
            cads_558_arr=cads_558_arr,
            labels_dict=labels_dict
        )

        # 4. Overwrite ATLAS/max_anatomy
        if "ATLAS/max_anatomy" in f:
            f["ATLAS/max_anatomy"][:] = merged_anat
        else:
            f.create_dataset("ATLAS/max_anatomy", data=merged_anat, compression="gzip", compression_opts=4, shuffle=True)

        # 5. Overwrite BASELINE/max_anatomy.nii.gz (CT baseline only)
        if "BASELINE/max_anatomy.nii.gz" in f:
            f["BASELINE/max_anatomy.nii.gz"][:] = merged_anat

        # NOTE: TFM_* (MRI follow-up) groups are intentionally NOT touched. MRI
        # timepoints own their max_anatomy (prostate zones); CT max_anatomy must
        # never be displayed over MRI.

        # 7. Update _meta_/max_anatomy_labels.json
        if "_meta_/max_anatomy_labels.json" in f:
            del f["_meta_/max_anatomy_labels.json"]
        f.create_dataset("_meta_/max_anatomy_labels.json", data=json.dumps(updated_labels))

    print(f"[SUCCESS] Patched {h5_path} with CADS structures. Updated labels: {len(updated_labels)}")
    return True


if __name__ == "__main__":
    print("CADS Anatomy Wiring Module loaded.")
    print("Configured CADS structures:")
    for lid, name in sorted(CADS_STRUCTURE_MAP.items()):
        print(f"  Label {lid:3d} -> {name}")
