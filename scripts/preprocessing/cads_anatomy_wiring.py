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


_EXCLUDED_EXACT_MASKS = frozenset({
    "body", "body_trunc", "body_extremities",
    "torso_fat", "subcutaneous_fat", "skin", "skeletal_muscle",
    "abdominal_cavity", "thoracic_cavity", "thorax_wall",
    "nasal_cavity_left", "nasal_cavity_right", "lung_airways_wall",
    "oblique_hyoid_plane", "digastric_plane_left", "digastric_plane_right",
})


def is_excluded_non_organ_mask(name: str) -> bool:
    """Return True for body-envelope, fat/skin/muscle, geometric plane, cavity, or helper masks."""
    ln = (name or "").strip().lower()
    if not ln:
        return True
    if ln in _EXCLUDED_EXACT_MASKS:
        return True
    if ln.endswith("_plane") or "_plane_" in ln:
        return True
    if any(bad in ln for bad in ("_temp", "helper", "dilation", "buffer", "_cavity", "chest_wall", "exclusion_zone")):
        return True
    if ln == "temp" or ln.startswith("temp_"):
        return True
    return False


def is_lung_label_name(name: str) -> bool:
    """Return True for lung structures (the ONLY non-LN anatomy allowed to override LN areas)."""
    ln = (name or "").strip().lower()
    if not ln or "airways_wall" in ln:
        return False
    return "lung" in ln


def is_lymph_label_name(name: str) -> bool:
    """Name-based lymph node area test (mirrors LesionAssociation.is_lymph_node_structure).

    Excludes non-lymph helper/structural labels that share station prefixes
    (e.g. 'abdominal_cavity', 'thoracic_cavity', 'thoracic_chest_wall',
    'thoracic_sternum_exclusion_zone', 'inguinal_ligament_*', '*_helper', '*_bridge').
    """
    ln = (name or "").lower()
    if not ln:
        return False
    if any(bad in ln for bad in ("_cavity", "exclusion_zone", "chest_wall", "_helper", "_bridge", "_base", "_dilated", "ligament")):
        return False
    if ln in ("thoracic_aorta", "thoracic_duct"):
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
                          labels_dict: dict = None,
                          flip_y_cads: bool = False) -> tuple:
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
      - When `flip_y_cads=True`, applies np.flip(..., axis=1) to the CADS arrays
        before merging so they match preflipped HDF5 volumes.

    Returns:
      (merged_anatomy_arr, updated_labels_dict)
    """
    if flip_y_cads:
        if cads_556_arr is not None:
            cads_556_arr = np.flip(cads_556_arr, axis=1)
        if cads_558_arr is not None:
            cads_558_arr = np.flip(cads_558_arr, axis=1)

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


def load_resampled_cads_arrays(case_dir: str, ref_img: sitk.Image = None) -> tuple:
    """Loads (cads_556_arr, cads_558_arr) for `case_dir`, resampled onto `ref_img` if given."""
    pred_dir = os.path.join(case_dir, "cads", "Fixed_CT_Volume")
    p556 = os.path.join(pred_dir, "Fixed_CT_Volume_part_556.nii.gz")
    p558 = os.path.join(pred_dir, "Fixed_CT_Volume_part_558.nii.gz")
    if ref_img is None:
        ref_ct_path = os.path.join(case_dir, "raw", "Fixed_CT_Volume.nii.gz")
        if os.path.isfile(ref_ct_path):
            ref_img = sitk.ReadImage(ref_ct_path)

    def _read(path):
        if not os.path.isfile(path):
            return None
        img = sitk.ReadImage(path)
        return resample_mask_to_reference(img, ref_img) if ref_img is not None else sitk.GetArrayFromImage(img)

    return _read(p556), _read(p558)


def classify_organ_priority(name: str) -> int:
    """Tissue priority for lesion naming (lower = higher priority)."""
    ln = (name or "").lower().strip()
    if not ln:
        return 6
    if ln in ("pz", "tz", "cz", "afs", "ur", "pg", "sv_l", "sv_r", "svl", "svr"):
        return 1
    if any(k in ln for k in ("prostate", "peripheral zone", "(pz)", "transition zone", "(tz)",
                             "central zone", "(cz)", "seminal_vesicle", "seminal vesicle", "seminal", "(sv_")):
        return 1
    if is_lymph_label_name(ln) or "station_" in ln:
        return 2
    bone_kw = ["vertebra", "rib_", "femur", "humerus", "scapula", "hip", "sacrum", "skull",
               "sternum", "clavicle", "costal", "hyoid", "mandible", "ilium", "ischium",
               "pubis", "tibia", "bone", "spine"]
    vessel_excl = ["artery", "vein", "vena", "vessel"]
    muscle_excl = ["levator", "subscapularis", "infraspinatus", "supraspinatus", "teres", "coracobrachial"]
    if any(k in ln for k in bone_kw) and not any(v in ln for v in vessel_excl) and not any(m in ln for m in muscle_excl):
        return 3
    organ_kw = ["liver", "kidney", "lung", "spleen", "pancreas", "heart", "thyroid",
                "adrenal", "stomach", "colon", "sigmoid", "rectum", "esophagus",
                "gallbladder", "bladder", "bowel", "duodenum", "trachea", "brain",
                "spinal_cord", "eye", "lacrimal", "parotid", "submandibular",
                "pituitary", "optic_nerve", "optic_chiasm", "seminal"]
    if any(k in ln for k in organ_kw):
        return 4
    if any(k in ln for k in ["aorta", "artery", "vein", "vena", "trunk", "carotid", "jugular"]):
        return 5
    return 6


_EXCLUDED_CONTAINER_LABELS = frozenset(
    ("body_trunc", "body_extremities", "abdominal_cavity", "mediastinum", "thoracic_cavity")
)


def pick_best_organ(counts: dict, labels_dict: dict) -> str:
    """Pick the best organ name from {label_id: voxel_count} matching Julia pick_best_organ."""
    filtered = {
        int(aid): int(cnt)
        for aid, cnt in counts.items()
        if cnt > 0 and labels_dict.get(str(int(aid)), "") not in ("", *_EXCLUDED_CONTAINER_LABELS)
    }
    if not filtered:
        return "Unknown"

    total_vox = sum(filtered.values())
    cands = []
    lac_cands = []
    for aid, cnt in filtered.items():
        aname = labels_dict[str(aid)]
        pri = classify_organ_priority(aname)
        cands.append((pri, -cnt, aname))
        if "lacrimal" in aname.lower():
            lac_cands.append((cnt, aname))

    # Rule 1 & 2 (Priority 1 Prostate/Seminal Vesicle, Priority 2 Lymph Node)
    top_pri_cands = [c for c in cands if c[0] in (1, 2)]
    if top_pri_cands:
        top_pri_cands.sort()
        return top_pri_cands[0][2]

    # Rule 2b: Lacrimal glands take priority over adjacent orbital skull/eye/skin
    if lac_cands:
        lac_cands.sort(reverse=True)
        top_lac_cnt, top_lac_name = lac_cands[0]
        if top_lac_cnt >= 2 and (top_lac_cnt >= total_vox * 0.015 or total_vox <= 250):
            return top_lac_name

    cands.sort()
    best_pri, neg_cnt, best_name = cands[0]
    best_cnt = -neg_cnt
    best_lc = best_name.lower()

    # Head/orbital gland & globe override over skull/mandible/calvarium
    if best_pri == 3 and any(k in best_lc for k in ("skull", "mandible", "calvarium")):
        head_kws = ("lacrimal", "parotid", "submandibular", "pituitary", "eye_left", "eye_right")
        gland_cands = [(-nc, nm) for _, nc, nm in cands if any(hk in nm.lower() for hk in head_kws)]
        if gland_cands:
            gland_cands.sort(reverse=True)
            g_cnt, g_name = gland_cands[0]
            if g_cnt > best_cnt or g_cnt >= total_vox * 0.15:
                return g_name

    if best_cnt < total_vox * 0.10 and len(cands) > 1:
        maj = max(cands, key=lambda x: -x[1])
        if -maj[1] > total_vox * 0.50:
            return maj[2]

    return best_name


def remove_edge_slice_lesions(lesion_arr: np.ndarray, edge_slices: int = 2) -> list:
    """Unconditionally zero out lesion voxels in the first/last `edge_slices` Z slices.

    Those outer axial slices frequently contain scanner end-of-FOV / truncation
    artifacts regardless of whether the semantic filter is enabled.
    Returns the sorted list of lesion IDs that had zero remaining voxels after trimming.
    """
    if lesion_arr is None or lesion_arr.ndim != 3 or edge_slices <= 0:
        return []
    z_dim = lesion_arr.shape[0]
    if z_dim <= 2 * edge_slices:
        return []

    edge_mask = np.zeros(z_dim, dtype=bool)
    edge_mask[:edge_slices] = True
    edge_mask[-edge_slices:] = True
    if not np.any(lesion_arr[edge_mask] > 0):
        return []

    before_ids = {int(x) for x in np.unique(lesion_arr) if x > 0}
    lesion_arr[:edge_slices, :, :] = 0
    lesion_arr[-edge_slices:, :, :] = 0
    after_ids = {int(x) for x in np.unique(lesion_arr) if x > 0}
    return sorted(before_ids - after_ids)


def filter_lacrimal_lesions(lesion_arr: np.ndarray,
                            max_anat_arr: np.ndarray,
                            labels_dict: dict,
                            seg_names: dict = None) -> tuple:
    """Filters physiological lacrimal gland uptake lesions from `lesion_arr` in place.

    Returns (removed_lids_list, remaining_lids_list, ct_organs_dict).
    `lesion_arr` and `max_anat_arr` are expected in matching (Z, Y, X) numpy order.
    """
    lac_ids = {int(k) for k, v in labels_dict.items() if "lacrimal" in v.lower()}
    eye_ids = {int(k) for k, v in labels_dict.items() if v.lower().strip() in ("eye_left", "eye_right", "eye_l", "eye_r")}
    orbital_allowed_ids = {
        int(k) for k, v in labels_dict.items()
        if "lacrimal" in v.lower()
        or v.lower().strip() in ("eye_left", "eye_right", "eye_l", "eye_r", "head", "skin", "face")
        or "skull" in v.lower()
    }

    gland_totals = {}
    if lac_ids and max_anat_arr is not None:
        u_m, c_m = np.unique(max_anat_arr[np.isin(max_anat_arr, list(lac_ids))], return_counts=True)
        for uv, uc in zip(u_m, c_m):
            gland_totals[int(uv)] = int(uc)
    has_lac_voxels = sum(gland_totals.values()) > 0

    all_lids = [int(i) for i in np.unique(lesion_arr) if i > 0]
    removed_lids = []
    remaining_lids = []
    ct_organs = {}
    z_dim = lesion_arr.shape[0]

    for lid in all_lids:
        mask_l = (lesion_arr == lid)
        n_vox = int(mask_l.sum())
        if n_vox <= 0:
            continue
        avals = max_anat_arr[mask_l]
        avals_nz = avals[avals > 0]
        counts = {}
        if len(avals_nz) > 0:
            uvals, ucnts = np.unique(avals_nz, return_counts=True)
            counts = {int(uv): int(uc) for uv, uc in zip(uvals, ucnts)}

        best_org = pick_best_organ(counts, labels_dict)
        if best_org == "Unknown" and seg_names is not None:
            sname = seg_names.get(str(lid), "").lower()
            if "prostate" in sname or "seminal" in sname:
                best_org = "Prostate"
            elif "lacrimal" in sname:
                best_org = "lacrimal_gland"
            elif any(bk in sname for bk in ("hip", "femur", "vertebra", "rib")):
                best_org = "Bone"
            elif "lung" in sname:
                best_org = "Lung"
            elif "liver" in sname:
                best_org = "Liver"

        is_lac = ("lacrimal" in best_org.lower())
        if not is_lac and lac_ids:
            for aid in lac_ids:
                ov = counts.get(aid, 0)
                if ov >= 5:
                    gtot = max(1, gland_totals.get(aid, 0))
                    if (ov / n_vox >= 0.08) or (gtot > 0 and ov / gtot >= 0.20):
                        is_lac = True
                        break

        if not is_lac and not has_lac_voxels and eye_ids:
            z_coords = np.where(mask_l)[0]
            mean_z = float(z_coords.mean()) if len(z_coords) > 0 else 0.0
            if z_dim > 0 and (mean_z / z_dim > 0.72):
                ov_eye = sum(counts.get(eid, 0) for eid in eye_ids)
                if ov_eye >= 30 and (ov_eye / n_vox >= 0.15):
                    known_cids = [aid for aid in counts if labels_dict.get(str(aid), "") not in ("", *_EXCLUDED_CONTAINER_LABELS)]
                    if known_cids and all(aid in orbital_allowed_ids for aid in known_cids):
                        is_lac = True

        if is_lac:
            lesion_arr[mask_l] = 0
            removed_lids.append(lid)
        else:
            remaining_lids.append(lid)
            ct_organs[str(lid)] = best_org

    return removed_lids, remaining_lids, ct_organs


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

    cads_558_arr = None
    if cads_558_path and os.path.exists(cads_558_path):
        img_558 = sitk.ReadImage(cads_558_path)
        if ref_img is not None:
            cads_558_arr = resample_mask_to_reference(img_558, ref_img)
        else:
            cads_558_arr = sitk.GetArrayFromImage(img_558)

    with h5py.File(h5_path, "r+") as f:
        is_preflipped = True
        if "_meta_/preflipped" in f:
            try:
                is_preflipped = int(f["_meta_/preflipped"][()]) == 1
            except Exception:
                is_preflipped = True

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

        # 3. Merge CADS structures (flipping CADS along Y iff HDF5 is preflipped)
        merged_anat, updated_labels = merge_cads_structures(
            base_anat,
            cads_556_arr=cads_556_arr,
            cads_558_arr=cads_558_arr,
            labels_dict=labels_dict,
            flip_y_cads=is_preflipped
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


def iter_authoritative_lymph_node_masks(ct_shape: tuple, raw_dir: str, ref_img: sitk.Image = None):
    """Yield (clean_name, bool_mask_zyx, source_tag) from authoritative lymph node segmentations.

    Priority:
      1. <case_dir>/lymph_nodes/ (or <raw_dir>/lymph_nodes/):
         Uses `all_lymph_nodes.nii.gz` + `lymph_node_labels.json` when present, plus
         any standalone `<station>.nii.gz` files not covered by the JSON map.
      2. Fallback to `<raw_dir>/final_results_gpu.h5`.
    """
    Z, Y, X = ct_shape
    case_dir = os.path.dirname(os.path.abspath(raw_dir))
    ln_dirs = [os.path.join(case_dir, "lymph_nodes"), os.path.join(raw_dir, "lymph_nodes")]
    ln_dir = next((d for d in ln_dirs if os.path.isdir(d)), None)

    if ln_dir is not None:
        all_nii = os.path.join(ln_dir, "all_lymph_nodes.nii.gz")
        lbl_json = os.path.join(ln_dir, "lymph_node_labels.json")
        seen_names = set()

        def _read_nii_mask(nii_p: str) -> np.ndarray:
            im = sitk.ReadImage(nii_p)
            if im.GetSize() != (X, Y, Z) and ref_img is not None:
                return resample_mask_to_reference(im, ref_img)
            return sitk.GetArrayFromImage(im)

        if os.path.isfile(all_nii) and os.path.isfile(lbl_json):
            try:
                with open(lbl_json, "r") as jf:
                    lbl_map = json.load(jf)
                arr_all = _read_nii_mask(all_nii)
                if arr_all.shape == ct_shape:
                    for lid_str, st_name in sorted(lbl_map.items(), key=lambda kv: str(kv[1]).lower()):
                        clean = str(st_name).strip().lower()
                        if not clean or not is_lymph_label_name(clean):
                            continue
                        seen_names.add(clean)
                        mask = (arr_all == int(lid_str))
                        if mask.any():
                            yield clean, mask, "lymph_nodes/all_lymph_nodes.nii.gz"
            except Exception as exc:
                print(f"    [WARN] Error reading {all_nii}: {exc}")

        for fn in sorted(os.listdir(ln_dir)):
            if not fn.endswith(".nii.gz") or fn == "all_lymph_nodes.nii.gz":
                continue
            clean = fn[:-7].strip().lower()
            if clean in seen_names or not is_lymph_label_name(clean):
                continue
            try:
                arr = _read_nii_mask(os.path.join(ln_dir, fn))
                if arr.shape == ct_shape:
                    mask = (arr > 0)
                    if mask.any():
                        seen_names.add(clean)
                        yield clean, mask, f"lymph_nodes/{fn}"
            except Exception as exc:
                print(f"    [WARN] Error reading {fn}: {exc}")
        return

    ln_h5 = os.path.join(raw_dir, "final_results_gpu.h5")
    if os.path.isfile(ln_h5):
        try:
            with h5py.File(ln_h5, "r") as lf:
                for k in sorted(lf.keys()):
                    clean = k.lower()
                    if not is_lymph_label_name(clean):
                        continue
                    arr_ln = lf[k][:]
                    if arr_ln.ndim == 3:
                        if arr_ln.shape == (X, Y, Z) and arr_ln.shape != (Z, Y, X):
                            arr_ln = np.transpose(arr_ln, (2, 1, 0))
                        if arr_ln.shape == ct_shape:
                            m_ln = (arr_ln > 0)
                            if m_ln.any():
                                yield clean, m_ln, "final_results_gpu.h5"
        except Exception as exc:
            print(f"    [WARN] Error reading {ln_h5}: {exc}")


def parse_seg_nrrd_names(ml_img: sitk.Image) -> dict:
    """Extract {str(label_value): segment_name} from Slicer .seg.nrrd metadata keys."""
    names = {}
    keys = set(ml_img.GetMetaDataKeys())
    prefixes = sorted({k.split("_")[0] for k in keys if k.startswith("Segment") and "_" in k})
    for pref in prefixes:
        lv_k = f"{pref}_LabelValue"
        nm_k = f"{pref}_Name"
        if lv_k in keys and nm_k in keys:
            try:
                lv = str(int(ml_img.GetMetaData(lv_k)))
                nm = str(ml_img.GetMetaData(nm_k)).strip()
                if nm:
                    names[lv] = nm
            except Exception:
                pass
    return names


def resample_mr_lesions_nrrd(ml_img: sitk.Image, ct_img: sitk.Image, tfm_obj: sitk.Transform = None) -> tuple:
    """Resample Slicer NNUNET_OR_Processed_Lesions_MR.seg.nrrd onto CT grid preserving distinct lesion IDs.

    Slicer exports cropped .seg.nrrd segmentations already in Fixed CT physical space (Identity);
    only fall back to `tfm_obj` if Identity maps outside `ct_img`.
    Returns (ml_arr_zyx_int16, mr_seg_names_dict).
    """
    res_id = sitk.Resample(ml_img, ct_img, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkInt16)
    ml_arr = sitk.GetArrayFromImage(res_id).astype(np.int16)
    if not ml_arr.any() and tfm_obj is not None:
        res_tfm = sitk.Resample(ml_img, ct_img, tfm_obj, sitk.sitkNearestNeighbor, 0, sitk.sitkInt16)
        ml_arr = sitk.GetArrayFromImage(res_tfm).astype(np.int16)
    raw_names = parse_seg_nrrd_names(ml_img)
    seg_names = {
        str(int(lid)): raw_names.get(str(int(lid)), f"Lesion {int(lid)}")
        for lid in np.unique(ml_arr)
        if int(lid) > 0
    }
    return ml_arr, seg_names


if __name__ == "__main__":
    print("CADS Anatomy Wiring Module loaded.")
    print("Configured CADS structures:")
    for lid, name in sorted(CADS_STRUCTURE_MAP.items()):
        print(f"  Label {lid:3d} -> {name}")
