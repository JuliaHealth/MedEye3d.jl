#!/usr/bin/env python3
"""
Fast in-place repair of CT lesions (PET_Lesions_0.nii.gz, CENTROIDS, organ_mapping,
segment_names.json, matches.json, scene_hierarchy.json) for Prostata_bimodal cases,
plus streaming upload to XNAT Subject MEDEYE3D.
"""
import os
import sys
import json
import time
import glob
import h5py
import numpy as np
import SimpleITK as sitk

sys.path.insert(0, "/workspaces/MedEye3d.jl/scripts/preprocessing")
from batch_process_cohort_dual_gpu import (
    XnatManager, PROJ_ID, OUT_BASE_DIR, CENSUS_PATH,
    download_ac_pet_dicom, calibrate_suv_image
)


def classify_organ_priority(name: str) -> int:
    ln = name.lower()
    if any(k in ln for k in ["prostate", "peripheral zone", "(pz)", "transition zone", "(tz)", "central zone", "(cz)", "seminal vesicle"]):
        return 1
    if any(k in ln for k in ["lymph", "knoten", "node", "station_"]) or any(ln.startswith(p) for p in ["neck_", "thoracic_", "axillary_", "abdominal_", "deep_inguinal", "superficial_inguinal", "inguinal_"]):
        return 2
    bone_kw = ["vertebra", "rib_", "femur", "humerus", "scapula", "hip", "sacrum", "skull", "sternum", "clavicle", "costal", "hyoid", "mandible", "ilium", "ischium", "pubis", "tibia", "bone", "spine"]
    vessel_excl = ["artery", "vein", "vena", "vessel"]
    if any(k in ln for k in bone_kw) and not any(v in ln for v in vessel_excl):
        return 3
    organ_kw = ["liver", "kidney", "lung", "spleen", "pancreas", "heart", "thyroid", "adrenal", "stomach", "colon", "rectum", "esophagus", "gallbladder", "bladder", "bowel", "duodenum", "trachea", "brain", "spinal_cord"]
    if any(k in ln for k in organ_kw):
        return 4
    if any(k in ln for k in ["aorta", "artery", "vein", "vena", "trunk", "carotid", "jugular"]):
        return 5
    return 6


def repair_case_lesions(pat_id: str, subj_info: dict, xnat: XnatManager, upload: bool = True) -> bool:
    case_dir = os.path.join(OUT_BASE_DIR, pat_id)
    raw_dir = os.path.join(case_dir, "raw")
    h5_path = os.path.join(case_dir, "preprocessed_volumes.h5")
    if not os.path.isfile(h5_path):
        return False

    ct_path = os.path.join(raw_dir, "Fixed_CT_Volume.nii.gz")
    seg_path = os.path.join(raw_dir, "NNUNET_OR_Processed_Lesions_CT.seg.nrrd")

    subj_id = subj_info["subj_id"]
    subj_label = subj_info["subj_label"]

    if not os.path.isfile(seg_path):
        sf_map = xnat.get_files_map(f"/data/projects/{PROJ_ID}/subjects/{subj_id}/resources/slicer_final/files?format=json")
        if "NNUNET_OR_Processed_Lesions_CT.seg.nrrd" in sf_map:
            os.makedirs(raw_dir, exist_ok=True)
            xnat.download_file(sf_map["NNUNET_OR_Processed_Lesions_CT.seg.nrrd"], seg_path)

    if not os.path.isfile(seg_path) or not os.path.isfile(ct_path):
        print(f"  [{pat_id}] Missing seg_path or ct_path, skipping.")
        return False

    ct_img = sitk.ReadImage(ct_path)
    cl_img = sitk.ReadImage(seg_path)

    ct_seg_names = {}
    for k in cl_img.GetMetaDataKeys():
        if k.startswith("Segment") and k.endswith("_Name"):
            prefix = k[:-5]
            val_k = f"{prefix}_LabelValue"
            if cl_img.HasMetaDataKey(val_k):
                lv = str(int(cl_img.GetMetaData(val_k)))
                ct_seg_names[lv] = cl_img.GetMetaData(k)

    if cl_img.GetNumberOfComponentsPerPixel() > 1:
        comb_arr = np.zeros(cl_img.GetSize()[::-1], dtype=np.int16)
        for k in cl_img.GetMetaDataKeys():
            if k.startswith("Segment") and k.endswith("_LabelValue"):
                prefix = k[:-11]
                lv = int(cl_img.GetMetaData(k))
                layer_idx = int(cl_img.GetMetaData(f"{prefix}_Layer")) if cl_img.HasMetaDataKey(f"{prefix}_Layer") else 0
                comp_img = sitk.VectorIndexSelectionCast(cl_img, layer_idx, sitk.sitkInt16)
                comp_arr = sitk.GetArrayFromImage(comp_img)
                comb_arr[comp_arr == lv] = lv
        comb_img = sitk.GetImageFromArray(comb_arr)
        comb_img.CopyInformation(cl_img)
        cl_img = comb_img

    res_cl = sitk.Resample(cl_img, ct_img, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkInt16)
    ct_lesion_arr = sitk.GetArrayFromImage(res_cl).astype(np.int16)
    ct_les_f = np.flip(ct_lesion_arr, axis=1)

    ct_labels = [int(i) for i in np.unique(ct_les_f) if i > 0]
    for i in ct_labels:
        if str(i) not in ct_seg_names:
            ct_seg_names[str(i)] = f"Lesion {i}"

    # Also verify/repair PET SUV if SUV_PET_Image_0.nii.gz was removed (due to broken Lu177 symlink)
    suv_path = os.path.join(raw_dir, "SUV_PET_Image_0.nii.gz")
    dcm_dir = os.path.join(raw_dir, "PET_AC_DICOM")
    uncorr_pet_dest = os.path.join(raw_dir, "SUV_PET_WB_120min_Uncorrected.nii.gz")
    pet_arr_f_new = None
    if not os.path.isfile(suv_path):
        exp_id = subj_info.get("exp_id", "")
        print(f"  [{pat_id}] Re-downloading genuine Prostata_bimodal AC PET DICOM and calibrating SUV...")
        download_ac_pet_dicom(xnat, exp_id, dcm_dir, pat_id=pat_id, subj_label=subj_label)
        suv_img = calibrate_suv_image(dcm_dir, ct_img, uncorr_pet_dest, suv_path)
        pet_arr = sitk.GetArrayFromImage(suv_img).astype(np.float32)
        pet_arr_f_new = np.flip(pet_arr, axis=1)

    with h5py.File(h5_path, "r+") as h5:
        sp_arr = np.array(ct_img.GetSpacing(), dtype=np.float64)
        orig_arr = np.array(ct_img.GetOrigin(), dtype=np.float64)
        dir_arr = np.array(ct_img.GetDirection(), dtype=np.float64)

        b = h5["BASELINE"]
        if pet_arr_f_new is not None:
            if "SUV_PET_Image_0.nii.gz" in b:
                del b["SUV_PET_Image_0.nii.gz"]
            ds_p0 = b.create_dataset("SUV_PET_Image_0.nii.gz", data=pet_arr_f_new, compression="gzip", compression_opts=4, shuffle=True)
            ds_p0.attrs["spacing"] = sp_arr
            ds_p0.attrs["origin"] = orig_arr
            ds_p0.attrs["direction"] = dir_arr
            for tk in [k for k in h5.keys() if k.startswith("TFM_Transform_FollowUp_to_Baseline_")]:
                tp_num = tk.replace("TFM_Transform_FollowUp_to_Baseline_", "").replace(".tfm", "")
                p_name = f"SUV_PET_Image_{tp_num}.nii.gz"
                if p_name in h5[tk]:
                    del h5[tk][p_name]
                ds_pt = h5[tk].create_dataset(p_name, data=pet_arr_f_new, compression="gzip", compression_opts=4, shuffle=True)
                ds_pt.attrs["spacing"] = sp_arr
                ds_pt.attrs["origin"] = orig_arr
                ds_pt.attrs["direction"] = dir_arr

        # 1. Update BASELINE/PET_Lesions_0.nii.gz
        if "PET_Lesions_0.nii.gz" in b:
            del b["PET_Lesions_0.nii.gz"]
        ds_l0 = b.create_dataset("PET_Lesions_0.nii.gz", data=ct_les_f, compression="gzip", compression_opts=4, shuffle=True)
        ds_l0.attrs["spacing"] = sp_arr
        ds_l0.attrs["origin"] = orig_arr
        ds_l0.attrs["direction"] = dir_arr

        # Read max_anatomy & anat_labels & mr_gold_f
        max_anat_f = h5["ATLAS/max_anatomy"][()]
        raw_al = h5["_meta_/max_anatomy_labels.json"][()]
        anat_labels = json.loads(raw_al.decode() if isinstance(raw_al, bytes) else raw_al)

        mr_gold_f = np.zeros_like(ct_les_f)
        if "TFM_Transform_FollowUp_to_Baseline_1.tfm" in h5 and "PET_Lesions_1.nii.gz" in h5["TFM_Transform_FollowUp_to_Baseline_1.tfm"]:
            mr_gold_f = h5["TFM_Transform_FollowUp_to_Baseline_1.tfm/PET_Lesions_1.nii.gz"][()]

        mr_seg_names = {"1": "Prostate Primary Lesion"} if mr_gold_f.any() else {}

        # 2. Update CENTROIDS
        if "CENTROIDS" in h5:
            del h5["CENTROIDS"]
        c_grp = h5.create_group("CENTROIDS")
        for lid in ct_labels:
            zc, yc, xc = np.mean(np.argwhere(ct_les_f == lid), axis=0)
            c_grp.create_dataset(f"tp0_lid{lid}", data=np.array([round(xc) + 1, round(yc) + 1, round(zc) + 1], dtype=np.int32))
        if mr_gold_f.any():
            zc, yc, xc = np.mean(np.argwhere(mr_gold_f == 1), axis=0)
            c_grp.create_dataset("tp1_lid1", data=np.array([round(xc) + 1, round(yc) + 1, round(zc) + 1], dtype=np.int32))

        # 3. Compute organ_mapping
        ct_organs = {}
        for lid in ct_labels:
            mask_l = (ct_les_f == lid)
            avals = max_anat_f[mask_l]
            avals = avals[avals > 0]
            best_org = "Unknown"
            if len(avals) > 0:
                uvals, ucnts = np.unique(avals, return_counts=True)
                cands = []
                for uv, uc in zip(uvals, ucnts):
                    aname = anat_labels.get(str(int(uv)), "")
                    if aname and aname not in ("body_trunc", "body_extremities", "abdominal_cavity", "mediastinum", "thoracic_cavity"):
                        cands.append((classify_organ_priority(aname), -int(uc), aname))
                if cands:
                    cands.sort()
                    best_org = cands[0][2]
            if best_org == "Unknown":
                sname = ct_seg_names.get(str(lid), "").lower()
                if "prostate" in sname:
                    best_org = "Prostate"
                elif "hip" in sname or "femur" in sname or "vertebra" in sname or "rib" in sname:
                    best_org = "Bone"
                elif "lung" in sname:
                    best_org = "Lung"
                elif "liver" in sname:
                    best_org = "Liver"
            ct_organs[str(lid)] = best_org

        tfm_keys = sorted([k for k in h5.keys() if k.startswith("TFM_Transform_FollowUp_to_Baseline_")])
        organ_mapping = {"0": ct_organs}
        segment_names_dict = {"0": ct_seg_names}
        for tk in tfm_keys:
            tp_num = tk.replace("TFM_Transform_FollowUp_to_Baseline_", "").replace(".tfm", "")
            organ_mapping[tp_num] = {"1": "Prostate"} if mr_gold_f.any() else {}
            segment_names_dict[tp_num] = mr_seg_names if mr_gold_f.any() else {}

        # 4. Compute matches.json
        matches = []
        best_ov_lid = None
        best_ov_cnt = 0
        if mr_gold_f.any() and ct_labels:
            for lid in ct_labels:
                ov = int(np.sum((mr_gold_f == 1) & (ct_les_f == lid)))
                if ov > best_ov_cnt:
                    best_ov_cnt = ov
                    best_ov_lid = lid
        for lid in ct_labels:
            children = []
            if mr_gold_f.any() and lid == best_ov_lid:
                children.append({
                    "name": "PET_Lesions_1",
                    "raw_lesion": "Segment_1",
                    "node": "child",
                    "group_id": int(lid),
                    "lesion": "Prostate Primary Lesion"
                })
            matches.append({
                "name": "PET_Lesions_0",
                "raw_lesion": f"Segment_{lid}",
                "node": "parent",
                "group_id": int(lid),
                "lesion": ct_seg_names.get(str(lid), f"Lesion {lid}"),
                "children": children
            })

        # 5. Update scene_hierarchy.json
        m_grp = h5["_meta_"]
        raw_sh = m_grp["scene_hierarchy.json"][()]
        scene_hierarchy = json.loads(raw_sh.decode() if isinstance(raw_sh, bytes) else raw_sh)
        for node in scene_hierarchy:
            if node.get("name") == "PET_Lesions_0":
                node["segments"] = [ct_seg_names.get(str(i), f"Lesion {i}") for i in ct_labels]

        for k_meta, val_str in [
            ("organ_mapping", json.dumps(organ_mapping)),
            ("segment_names.json", json.dumps(segment_names_dict)),
            ("matches.json", json.dumps(matches)),
            ("scene_hierarchy.json", json.dumps(scene_hierarchy))
        ]:
            if k_meta in m_grp:
                del m_grp[k_meta]
            m_grp.create_dataset(k_meta, data=val_str)

    # Repack to reclaim space if needed, or upload directly
    print(f"  [{pat_id}] Patched {len(ct_labels)} CT lesions ({(ct_les_f > 0).sum()} voxels) in {h5_path}")

    if upload:
        remote_h5 = f"{subj_label}_preprocessed.h5"
        xnat.request("PUT", f"/data/projects/{PROJ_ID}/subjects/{subj_id}/resources/MEDEYE3D")
        subj_h5_uri = f"/data/projects/{PROJ_ID}/subjects/{subj_id}/resources/MEDEYE3D/files/{remote_h5}?overwrite=true"
        t0 = time.time()
        ok = xnat.upload_file_streaming(subj_h5_uri, h5_path)
        print(f"  [{pat_id}] Uploaded {remote_h5} to XNAT in {time.time() - t0:.1f}s (ok={ok})")
    return True


if __name__ == "__main__":
    import re
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("cases", nargs="*", help="Case IDs (e.g. Pat9 Pat4)")
    parser.add_argument("--no-upload", action="store_true", help="Patch locally without uploading")
    parser.add_argument("--only-missing", action="store_true", help="Only patch/upload if PET_Lesions_0 max == 0")
    args = parser.parse_args()

    xnat = XnatManager()
    with open(CENSUS_PATH, "r") as cf:
        subj_map = json.load(cf)

    targets = args.cases
    if not targets:
        all_pids = sorted(subj_map.keys(), key=lambda x: int(re.search(r"\d+", x).group()) if re.search(r"\d+", x) else 9999)
        for pid in all_pids:
            h5_p = os.path.join(OUT_BASE_DIR, pid, "preprocessed_volumes.h5")
            if os.path.isfile(h5_p):
                if args.only_missing:
                    try:
                        with h5py.File(h5_p, "r") as f:
                            if int(f["BASELINE/PET_Lesions_0.nii.gz"][()].max()) > 0:
                                continue
                    except Exception:
                        pass
                targets.append(pid)

    print(f"Repairing {len(targets)} cases (upload={not args.no_upload})...")
    for pid in targets:
        if pid in subj_map:
            try:
                repair_case_lesions(pid, subj_map[pid], xnat, upload=not args.no_upload)
            except Exception as e:
                print(f"  [{pid}] Error during repair: {e}")
