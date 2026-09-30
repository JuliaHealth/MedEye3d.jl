import nibabel as nib
#!/usr/bin/env python3


"""
export_clean_mrb.py

Exports a clean 3D Slicer .mrb scene directly from:
  1) primary_masks.h5 (CT volume and metadata - NO NIfTI dependencies)
  2) final_results_gpu.h5 (Pure GPU-computed lymph nodes and helper masks)


Ensures that All_Lymph_Node_Areas.seg.nrrd contains ONLY pure clinical lymph node stations,
with zero label arithmetic collisions, proper colors, and clean layer organization.


"""
import os
import sys
import argparse
import tempfile
import zipfile
import h5py
import nrrd
import numpy as np




EXTERNAL_MODEL_LABEL_MAPS = {
    "hnlnl": {
        1: "Level_Ia_Submental", 2: "Level_Ib_Submandibular_Left",
        3: "Level_Ib_Submandibular_Right", 4: "Level_IIa_Left",
        5: "Level_IIa_Right", 6: "Level_IIb_Left", 7: "Level_IIb_Right",
        8: "Level_III_Left", 9: "Level_III_Right",
        10: "Level_IVa_Left", 11: "Level_IVa_Right",
        12: "Level_IVb_Left", 13: "Level_IVb_Right",
        14: "Level_Va_Left", 15: "Level_Va_Right",
        16: "Level_Vb_Left", 17: "Level_Vb_Right",
        18: "Level_VIa", 19: "Level_VIb", 20: "Level_VII"
    },
    "ovseg": {
        1: "spleen", 2: "kidney_right", 3: "kidney_left", 4: "gallbladder",
        5: "esophagus", 6: "liver", 7: "stomach", 8: "aorta",
        9: "inferior_vena_cava", 10: "pancreas", 11: "adrenal_right",
        12: "adrenal_left", 13: "duodenum", 14: "urinary_bladder",
        15: "prostate_uterus", 16: "rectum", 17: "small_bowel",
        18: "lymph_nodes"
    }
}


# Canonical 67 clinical lymph node stations in anatomical order
CLINICAL_LN_STATIONS = [
    # Neck
    "Neck_Level_Ia_Submental",
    "Neck_Level_Ib_Submandibular_Left",
    "Neck_Level_Ib_Submandibular_Right",
    "Neck_Level_IIa_Upper_Jugular_Left",
    "Neck_Level_IIa_Upper_Jugular_Right",
    "Neck_Level_IIb_Upper_Jugular_Left",
    "Neck_Level_IIb_Upper_Jugular_Right",
    "Neck_Level_III_Middle_Jugular_Left",
    "Neck_Level_III_Middle_Jugular_Right",
    "Neck_Level_IV_Lower_Jugular_Left",
    "Neck_Level_IV_Lower_Jugular_Right",
    "Neck_Level_Va_Upper_Posterior_Triangle_Left",
    "Neck_Level_Va_Upper_Posterior_Triangle_Right",
    "Neck_Level_VI_Anterior_Cervical",
    "Neck_Level_Xb_Occipital_Left",
    "Neck_Level_Xb_Occipital_Right",
    "Neck_Parotid_Left",
    "Neck_Parotid_Right",
    "Neck_Retropharyngeal",

    # Thoracic & Axillary
    "Thoracic_Station_2_UpperParatracheal_Left",
    "Thoracic_Station_2_UpperParatracheal_Right",
    "Thoracic_Station_3A_Prevascular_Left",
    "Thoracic_Station_3A_Prevascular_Right",
    "Thoracic_Station_3P_Retrotracheal_Left",
    "Thoracic_Station_3P_Retrotracheal_Right",
    "Thoracic_Station_4_LowerParatracheal_Left",
    "Thoracic_Station_4_LowerParatracheal_Right",
    "Thoracic_Station_5_Subaortic_Left",
    "Thoracic_Station_6_Paraaortic",
    "Thoracic_Station_7_Subcarinial",
    "Thoracic_Station_8_Paraoesophageal_Left",
    "Thoracic_Station_8_Paraoesophageal_Right",
    "Thoracic_Station_Hilar_Interlobar_Left",
    "Thoracic_Station_Hilar_Interlobar_Right",
    "Thoracic_Station_Prepericardial_Left",
    "Thoracic_Station_Prepericardial_Right",
    "Thoracic_Mammary_Left",
    "Thoracic_Mammary_Right",
    "Axillary_Level_I_Left",
    "Axillary_Level_I_Right",
    "Axillary_Level_II_Left",
    "Axillary_Level_II_Right",
    "Axillary_Level_III_Left",
    "Axillary_Level_III_Right",
    "Axillary_Rotter_Left",
    "Axillary_Rotter_Right",

    # Abdomen & Pelvis
    "Abdominal_Station_1_Right_Paracardial",
    "Abdominal_Station_2_Left_Paracardial",
    "Abdominal_Station_3_Lesser_Curvature",
    "Abdominal_Station_4_Greater_Curvature",
    "Abdominal_Station_5_Suprapyloric",
    "Abdominal_Station_6_Infrapyloric",
    "Abdominal_Station_7_Left_Gastric",
    "Abdominal_Station_8_Common_Hepatic",
    "Abdominal_Station_9_Celiac",
    "Abdominal_Station_10_Splenic_Hilum",
    "Abdominal_Station_11_Splenic_Artery",
    "Abdominal_Station_13_Posterior_Pancreaticoduodenal",
    "Abdominal_Station_17_Anterior_Pancreaticoduodenal",
    "Abdominal_Station_14_SMA",
    "Abdominal_Station_Inferior_Pancreatic",
    "Abdominal_Renal_Hilar_Left",
    "Abdominal_Renal_Hilar_Right",
    "Abdominal_Common_Iliac_Left",
    "Abdominal_Common_Iliac_Right",
    "Abdominal_External_Iliac_Left",
    "Abdominal_External_Iliac_Right",
    "Abdominal_Iliac_Bifurcation_Left",
    "Abdominal_Iliac_Bifurcation_Right",
    "Abdominal_Internal_Iliac_Left",
    "Abdominal_Internal_Iliac_Right",
    "Abdominal_Mesenteric_Interenteric",
    "Abdominal_Obturator_Left",
    "Abdominal_Obturator_Right",
    "Abdominal_Station_16a1_Aortic_Hiatus",
    "Abdominal_Station_16a2_Upper_Middle_Paraaortic",
    "Abdominal_Station_16b1_Lower_Middle_Paraaortic",
    "Abdominal_Station_16b2_Caudal_Paraaortic",
    "Abdominal_Pararectal",
    "Abdominal_Presacral",
    "Deep_Inguinal_Left",
    "Deep_Inguinal_Right",
    "Superficial_Inguinal_Left",
    "Superficial_Inguinal_Right"
]

import random

def get_station_color(name, idx):
    # Use exact same random seeding as old python engine for identical colors
    random.seed(idx + 42)
    r, g, b = random.random(), random.random(), random.random()
    return f"{r:.3f} {g:.3f} {b:.3f}"


def build_mrb(h5_primary_path, h5_results_path, output_mrb_path):
    print(f"Building Clean MRB:")
    print(f"  Primary H5: {h5_primary_path}")
    print(f"  Results H5: {h5_results_path}")
    print(f"  Output MRB: {output_mrb_path}")

    with h5py.File(h5_primary_path, "r") as f_prim, h5py.File(h5_results_path, "r") as f_res:
        meta_g = f_prim["metadata"].attrs
        dims = tuple(meta_g["dimensions"])
        spacing = tuple(meta_g["spacing"])
        origin = tuple(meta_g["origin"])
        case_id = meta_g.get("case_id", "Patient")
        if isinstance(case_id, bytes):
            case_id = case_id.decode("utf-8")
        elif hasattr(case_id, "tobytes"):
            case_id = case_id.tobytes().decode("utf-8", errors="ignore").rstrip("\x00")
        case_id = str(case_id)

        print(f"  Case ID: {case_id}, Dimensions: {dims}, Spacing: {spacing}")

        import tempfile
        tmpdir_obj = tempfile.TemporaryDirectory()
        tmpdir = tmpdir_obj.name
        bundle_name = f"{case_id}_Combined"
        bundle_dir = os.path.join(tmpdir, bundle_name)
        data_dir = os.path.join(bundle_dir, "Data")
        os.makedirs(data_dir, exist_ok=True)

        ct_raw = f_prim["ct/volume"][:]
        if ct_raw.shape == (dims[2], dims[1], dims[0]):
            ct_arr = np.transpose(ct_raw, (2, 1, 0))
        else:
            ct_arr = ct_raw

        nrrd_header_base = {
            'type': 'short',
            'dimension': 3,
            'space': 'left-posterior-superior',
            'sizes': np.array([dims[0], dims[1], dims[2]]),
            'space directions': np.array([
                [spacing[0], 0.0, 0.0],
                [0.0, spacing[1], 0.0],
                [0.0, 0.0, spacing[2]]
            ]),
            'space origin': np.array([origin[0], origin[1], origin[2]]),
            'endian': 'little',
            'encoding': 'gzip'
        }

        # Lymph Nodes ONLY into All_Lymph_Node_Areas.seg.nrrd
        ln_arr = np.zeros((dims[2], dims[1], dims[0]), dtype=np.uint16)
        ln_header = dict(nrrd_header_base)
        ln_header['type'] = 'unsigned short'
        ln_header['Segmentation_MasterRepresentation'] = 'Binary labelmap'

        res_keys_lower = {k.lower(): k for k in f_res.keys()}
        
        label_counter = 1
        seg_count = 0

        for idx, station in enumerate(CLINICAL_LN_STATIONS):
            st_lower = station.lower()
            mask_arr = None

            if st_lower in res_keys_lower:
                m_raw = f_res[res_keys_lower[st_lower]][:]
                mask_arr = m_raw

            if mask_arr is not None:
                fg = mask_arr > 0
                cnt = fg.sum()
                if cnt > 0:
                    ln_arr[fg] = label_counter

                    tag = f"Segment{seg_count}"
                    ln_header[f"{tag}_ID"] = f"Segment_{label_counter}"
                    ln_header[f"{tag}_Name"] = station
                    ln_header[f"{tag}_LabelValue"] = str(label_counter)
                    ln_header[f"{tag}_Layer"] = "0"
                    ln_header[f"{tag}_Color"] = get_station_color(station, label_counter)
                    ln_header[f"{tag}_ColorAutoGenerated"] = "1"
                    ln_header[f"{tag}_NameAutoGenerated"] = "0"


                    nz = np.nonzero(ln_arr == label_counter)
                    if len(nz[0]) > 0:
                        extent_str = f"{nz[2].min()} {nz[2].max()} {nz[1].min()} {nz[1].max()} {nz[0].min()} {nz[0].max()}"
                        ln_header[f"{tag}_Extent"] = extent_str
                    print(f"    [LN {seg_count+1}] {station}: {cnt} voxels (label {label_counter})")
                    label_counter += 1
                    seg_count += 1
                else:
                    print(f"    [LN EMPTY] {station}: 0 voxels")
            else:
                print(f"    [LN MISSING] {station}")

        print(f"  Total packed Lymph Node segments: {seg_count}")


        # Helpers into All_Helpers.seg.nrrd
        helper_arr = np.zeros((dims[2], dims[1], dims[0]), dtype=np.uint16)
        helper_header = dict(nrrd_header_base)
        helper_header['type'] = 'unsigned short'
        helper_header['Segmentation_MasterRepresentation'] = 'Binary labelmap'
        h_counter = 1
        h_seg_count = 0
        
        # Deterministic unique color per mask name (used by helpers and rest organs)
        import hashlib
        def get_color(name):
            h = int(hashlib.md5(name.encode('utf-8')).hexdigest(), 16)
            r = (h & 0xFF) / 255.0
            g = ((h >> 8) & 0xFF) / 255.0
            b = ((h >> 16) & 0xFF) / 255.0
            return f"{r*0.6+0.2:.3f} {g*0.6+0.2:.3f} {b*0.6+0.2:.3f}"
        
        # Masks to skip in All_Rest (too large / not useful)
        SKIP_REST = {"tissue", "body", "body_trunc", "body_extremities", "skin", "body_trunc_body", "tissue_muscle"}
        def is_derived_structure(name):
            kl = name.lower()
            return (kl.startswith("helper_") or "_helper_" in kl or 
                kl in ["thoracic_sternum_bridge", "thoracic_sternum_exclusion_zone"] or
                kl.endswith("_plane") or kl.endswith("_plane_left") or kl.endswith("_plane_right") or 
                "_proxy" in kl or "_computed" in kl or kl.startswith("plane_") or 
                kl.endswith("_point_left") or kl.endswith("_point_right") or "line" in kl)

        for k in sorted(f_res.keys()):
            if is_derived_structure(k):
                m_raw = f_res[k][:]
                m_arr = m_raw
                fg = m_arr > 0
                if fg.sum() > 0:
                    helper_arr[fg] = h_counter
                    tag = f"Segment{h_seg_count}"
                    helper_header[f"{tag}_ID"] = f"Segment_{h_counter}"
                    helper_header[f"{tag}_Name"] = k
                    helper_header[f"{tag}_LabelValue"] = str(h_counter)
                    helper_header[f"{tag}_Layer"] = "0"
                    helper_header[f"{tag}_Color"] = get_color(k)
                    h_counter += 1
                    h_seg_count += 1
                    
        if "masks" in f_prim:
            prim_masks = f_prim["masks"]
            for k in sorted(prim_masks.keys()):
                if is_derived_structure(k):
                    m_raw = prim_masks[k][:]
                    m_arr = m_raw
                    fg = m_arr > 0
                    if fg.sum() > 0:
                        helper_arr[fg] = h_counter
                        tag = f"Segment{h_seg_count}"
                        helper_header[f"{tag}_ID"] = f"Segment_{h_counter}"
                        helper_header[f"{tag}_Name"] = k
                        helper_header[f"{tag}_LabelValue"] = str(h_counter)
                        helper_header[f"{tag}_Layer"] = "0"
                        helper_header[f"{tag}_Color"] = get_color(k)
                        h_counter += 1
                        h_seg_count += 1
        print(f"  Total packed Helper segments: {h_seg_count}")

        # New Helpers (stylohyoid and sternohyoid) into new_helpers.seg.nrrd
        new_helpers_arr = np.zeros((dims[0], dims[1], dims[2]), dtype=np.uint16)
        new_helpers_header = dict(nrrd_header_base)
        new_helpers_header['type'] = 'unsigned short'
        new_helpers_header['Segmentation_MasterRepresentation'] = 'Binary labelmap'
        nh_counter = 1
        nh_seg_count = 0
        

        target_helper_planes = [
            "helper_digastric_proxy_left",
            "helper_digastric_proxy_right",
            "helper_digastric_proxy",
            "helper_stylohyoid_proxy_left",
            "helper_stylohyoid_proxy_right",
            "helper_stylohyoid_proxy",
            "helper_sternohyoid_proxy_left",
            "helper_sternohyoid_proxy_right",
            "helper_sternohyoid_proxy",
            "helper_jugular_anterior_proxy_left",
            "helper_jugular_anterior_proxy_right",
            "helper_jugular_anterior_proxy"
        ]
        
        helper_planes_arr = np.zeros((dims[2], dims[1], dims[0]), dtype=np.uint16)
        helper_planes_header = dict(nrrd_header_base)
        helper_planes_header['type'] = 'unsigned short'
        helper_planes_header['Segmentation_MasterRepresentation'] = 'Binary labelmap'
        hp_counter = 1
        hp_seg_count = 0
        
        for k in target_helper_planes:
            k_lower = k.lower()
            if k_lower in res_keys_lower:
                m_raw = f_res[res_keys_lower[k_lower]][:]
                m_arr = m_raw
                fg = m_arr > 0
                if fg.sum() > 0:
                    helper_planes_arr[fg] = hp_counter
                    tag = f"Segment{hp_seg_count}"
                    helper_planes_header[f"{tag}_ID"] = f"Segment_{hp_counter}"
                    helper_planes_header[f"{tag}_Name"] = k
                    helper_planes_header[f"{tag}_LabelValue"] = str(hp_counter)
                    helper_planes_header[f"{tag}_Layer"] = "0"
                    helper_planes_header[f"{tag}_Color"] = get_color(k)
                    hp_counter += 1
                    hp_seg_count += 1
        print(f"  Total packed Helper Planes segments: {hp_seg_count}")

        # Vessels into Vessels.seg.nrrd
        vessels_arr = np.zeros((dims[2], dims[1], dims[0]), dtype=np.uint16)
        vessels_header = dict(nrrd_header_base)
        vessels_header['type'] = 'unsigned short'
        vessels_header['Segmentation_MasterRepresentation'] = 'Binary labelmap'
        v_counter = 1
        v_seg_count = 0
        
        target_vessels = [
            "aorta", "inferior_vena_cava", "portal_vein_and_splenic_vein",
            "portal_vein", "splenic_vein",
            "common_hepatic_artery", "splenic_artery", "celiac_trunk", 
            "superior_mesenteric_artery", "inferior_mesenteric_artery", 
            "renal_artery_left", "renal_artery_right", 
            "renal_vein_left", "renal_vein_right", 
            "iliac_artery_common_left", "iliac_artery_common_right", 
            "iliac_vein_common_left", "iliac_vein_common_right", 
            "iliac_artery_external_left", "iliac_artery_external_right", 
            "iliac_vein_external_left", "iliac_vein_external_right", 
            "iliac_artery_internal_left", "iliac_artery_internal_right", 
            "iliac_vein_internal_left", "iliac_vein_internal_right", 
            "pulmonary_artery", "pulmonary_vein", "superior_vena_cava", 
            "brachiocephalic_vein_left", "brachiocephalic_vein_right", 
            "subclavian_artery_left", "subclavian_artery_right", 
            "subclavian_vein_left", "subclavian_vein_right", 
            "common_carotid_artery_left", "common_carotid_artery_right", 
            "internal_jugular_vein_left", "internal_jugular_vein_right"
        ]
        
        for k in target_vessels:
            k_lower = k.lower()
            if k_lower in res_keys_lower:
                m_raw = f_res[res_keys_lower[k_lower]][:]
                m_arr = m_raw
                fg = m_arr > 0
                if fg.sum() > 0:
                    vessels_arr[fg] = v_counter
                    tag = f"Segment{v_seg_count}"
                    vessels_header[f"{tag}_ID"] = f"Segment_{v_counter}"
                    vessels_header[f"{tag}_Name"] = k
                    vessels_header[f"{tag}_LabelValue"] = str(v_counter)
                    vessels_header[f"{tag}_Layer"] = "0"
                    vessels_header[f"{tag}_Color"] = get_color(k)
                    v_counter += 1
                    v_seg_count += 1
            elif "masks" in f_prim and k_lower in [mk.lower() for mk in f_prim["masks"].keys()]:
                prim_keys_lower = {mk.lower(): mk for mk in f_prim["masks"].keys()}
                m_raw = f_prim["masks"][prim_keys_lower[k_lower]][:]
                m_arr = m_raw
                fg = m_arr > 0
                if fg.sum() > 0:
                    vessels_arr[fg] = v_counter
                    tag = f"Segment{v_seg_count}"
                    vessels_header[f"{tag}_ID"] = f"Segment_{v_counter}"
                    vessels_header[f"{tag}_Name"] = k
                    vessels_header[f"{tag}_LabelValue"] = str(v_counter)
                    vessels_header[f"{tag}_Layer"] = "0"
                    vessels_header[f"{tag}_Color"] = get_color(k)
                    v_counter += 1
                    v_seg_count += 1
                    
        print(f"  Total packed Vessels segments: {v_seg_count}")


        # Primary Organs / Rest into All_Rest.seg.nrrd
        layers = [np.zeros((dims[2], dims[1], dims[0]), dtype=np.uint16)]
        rest_header = dict(nrrd_header_base)
        rest_header['type'] = 'unsigned short'
        rest_header['Segmentation_MasterRepresentation'] = 'Binary labelmap'
        r_counter = 1
        r_seg_count = 0

        if "masks" in f_prim:
            prim_masks = f_prim["masks"]
            # Exclude already exported LN stations or helpers if needed, but the primary_masks 
            # usually only contains TotalSegmentator/clinical structures, not our LNs.
            orgs_to_export = list(prim_masks.keys())
            
            
            for org in orgs_to_export:
                if org in SKIP_REST:
                    continue
                if org in prim_masks and not is_derived_structure(org):
                    m_raw = prim_masks[org][:]
                    m_arr = m_raw
                    fg = m_arr > 0
                    if fg.sum() > 0:
                        assigned_layer = -1
                        idx_1d = np.flatnonzero(fg)
                        if len(idx_1d) > 0:
                            if 'rest_bitmask' not in locals():
                                rest_bitmask = np.zeros(dims[0]*dims[1]*dims[2], dtype=np.uint64)
                            
                            used_bits = np.bitwise_or.reduce(rest_bitmask[idx_1d])
                            for i in range(60):
                                if not (used_bits & (np.uint64(1) << np.uint64(i))):
                                    assigned_layer = i
                                    break
                                    
                            rest_bitmask[idx_1d] |= (np.uint64(1) << np.uint64(assigned_layer))
                            
                        if assigned_layer >= len(layers):
                            for _ in range(assigned_layer - len(layers) + 1):
                                layers.append(np.zeros((dims[2], dims[1], dims[0]), dtype=np.uint16))
                                
                        layers[assigned_layer].ravel()[idx_1d] = r_counter
                        tag = f"Segment{r_seg_count}"
                        rest_header[f"{tag}_ID"] = f"Segment_{r_counter}"
                        rest_header[f"{tag}_Name"] = org
                        rest_header[f"{tag}_LabelValue"] = str(r_counter)
                        rest_header[f"{tag}_Layer"] = str(assigned_layer)
                        rest_header[f"{tag}_Color"] = get_color(org)

                        nz = np.nonzero(fg)
                        if len(nz[0]) > 0:
                            extent_str = f"{nz[2].min()} {nz[2].max()} {nz[1].min()} {nz[1].max()} {nz[0].min()} {nz[0].max()}"
                            rest_header[f"{tag}_Extent"] = extent_str
                        r_counter += 1
                        r_seg_count += 1

        print(f"  Total packed Rest Organ segments: {r_seg_count} across {len(layers)} layers")
        if len(layers) > 1:
            rest_arr = np.transpose(np.stack(layers, axis=-1), (2, 1, 0, 3))
            rest_header['dimension'] = 4
            rest_header['kinds'] = ['domain', 'domain', 'domain', 'list']
            # We must append [NaN, NaN, NaN] to space directions
            sd = list(rest_header['space directions'])
            sd.append([np.nan, np.nan, np.nan])
            rest_header['space directions'] = np.array(sd)
            if 'sizes' in rest_header: del rest_header['sizes']
        else:
            rest_arr = np.transpose(layers[0], (2, 1, 0))

        # --- EXTERNAL MODELS ---
        
        # --- DYNAMIC EXTERNAL MODELS SCANNER ---
        external_models_dir = os.path.join(os.path.dirname(args.h5), "external_model_segmentations")
        if not os.path.exists(external_models_dir):
            external_models_dir = "data/external_model_segmentations"
            
        external_model_nodes = []
        if os.path.exists(external_models_dir):
            import glob
            for model_name in os.listdir(external_models_dir):
                model_dir = os.path.join(external_models_dir, model_name)
                if not os.path.isdir(model_dir) or model_name.endswith("_in"): continue
                
                nifti_files = glob.glob(os.path.join(model_dir, "**", "*.nii.gz"), recursive=True)
                if not nifti_files: continue
                
                print(f"  Processing external model: {model_name} ({len(nifti_files)} files)")
                
                # We will pack all non-overlapping segmentations into a single 3D array for this model
                model_arr = np.zeros((dims[0], dims[1], dims[2]), dtype=np.uint16)
                model_header = dict(nrrd_header_base)
                model_header['type'] = 'unsigned short'
                model_header['Segmentation_MasterRepresentation'] = 'Binary labelmap'
                
                current_label = 1
                for nii_path in nifti_files:
                    basename = os.path.basename(nii_path).replace(".nii.gz", "")
                    
                    # Exclude body or cavities
                    exclude_keywords = ["body", "torso", "skin", "cavity", "fat", "subcutaneous", "patient", "background"]
                    if any(basename.lower() == kw or basename.lower().startswith(kw + "_") or basename.lower().endswith("_" + kw) for kw in exclude_keywords):
                        continue
                        
                    try:
                        import nibabel as nib
                        img = nib.load(nii_path)
                        raw_data = img.get_fdata()
                        # Handle probability maps (float data with max <= 1.0)
                        if raw_data.dtype in (np.float32, np.float64) and raw_data.max() <= 1.0:
                            data = np.ascontiguousarray((raw_data > 0.5).astype(np.uint16))
                        else:
                            data = np.ascontiguousarray(raw_data.astype(np.uint16))
                        if data.shape != tuple(dims):
                            print(f"    Skipping {nii_path} (shape {data.shape} != {dims})")
                            continue
                            
                        unique_vals = np.unique(data)
                        if len(unique_vals) > 200:
                            print(f"    Skipping {nii_path} (too many unique values: {len(unique_vals)}, likely raw image)")
                            continue
                        for val in unique_vals:
                            if val == 0: continue
                            
                            fg = (data == val)
                            non_overlapping_mask = fg & (model_arr == 0)
                            if non_overlapping_mask.sum() > 100:  # Skip noise segments with < 100 voxels
                                model_arr[non_overlapping_mask] = current_label
                                 
                                nz = np.nonzero(non_overlapping_mask)
                                extent_str = f"{nz[2].min()} {nz[2].max()} {nz[1].min()} {nz[1].max()} {nz[0].min()} {nz[0].max()}"

                                tag = f"Segment{current_label-1}"
                                model_header[f"{tag}_ID"] = f"Segment_{current_label}"
                                label_map = EXTERNAL_MODEL_LABEL_MAPS.get(model_name, {})
                                seg_name = label_map.get(int(val), f"{basename}_{int(val)}" if len(unique_vals) > 2 else basename)
                                model_header[f"{tag}_Name"] = seg_name
                                model_header[f"{tag}_LabelValue"] = str(current_label)
                                model_header[f"{tag}_Layer"] = "0"
                                model_header[f"{tag}_Color"] = get_color(f"{basename}_{int(val)}")
                                model_header[f"{tag}_Extent"] = extent_str
                                current_label += 1
                    except Exception as e:
                        print(f"    Failed reading {nii_path}: {e}")
                
                if current_label > 1:
                    out_nrrd_path = os.path.join(bundle_dir, "Data", f"{model_name}.seg.nrrd")
                    nrrd.write(out_nrrd_path, model_arr, model_header, index_order='F')
                    external_model_nodes.append(model_name)
                    print(f"  Saved {model_name}.seg.nrrd with {current_label-1} segments")

        # Slicer MRML scene
        bundle_name = f"{case_id}_Combined"
        mrml_content = f'''<?xml version="1.0" encoding="utf-8"?>
<MRML version="4.11">
  <ScalarVolumeDisplay id="vtkMRMLScalarVolumeDisplayNode1" window="400" level="40" autoWindowLevel="0" colorNodeID="vtkMRMLColorTableNodeGrey" visibility="true"/>
  <Volume id="vtkMRMLScalarVolumeNode1" name="CT" hideFromCameras="0" selected="true" storageNodeRef="vtkMRMLVolumeArchetypeStorageNode1" displayNodeRef="vtkMRMLScalarVolumeDisplayNode1" />
  <VolumeArchetypeStorage id="vtkMRMLVolumeArchetypeStorageNode1" fileName="Data/CT.nrrd" useCompression="1" centerImage="0" />
  <SliceComposite id="vtkMRMLSliceCompositeNodeRed" layoutName="Red" backgroundVolumeID="vtkMRMLScalarVolumeNode1" linkedControl="1" />
  <SliceComposite id="vtkMRMLSliceCompositeNodeYellow" layoutName="Yellow" backgroundVolumeID="vtkMRMLScalarVolumeNode1" linkedControl="1" />
  <SliceComposite id="vtkMRMLSliceCompositeNodeGreen" layoutName="Green" backgroundVolumeID="vtkMRMLScalarVolumeNode1" linkedControl="1" />

  <SegmentationDisplay id="vtkMRMLSegmentationDisplayNode_LNs" opacity="0.45" visibility="true" visibility2D="true" visibility3D="true" Visibility2DFill="true" Visibility2DOutline="true" />
  <Segmentation id="vtkMRMLSegmentationNode_LNs" name="All_Lymph_Node_Areas" storageNodeRef="vtkMRMLSegmentationStorageNode_LNs" displayNodeRef="vtkMRMLSegmentationDisplayNode_LNs" referenceImageGeometryReference="vtkMRMLScalarVolumeNode1" />
  <SegmentationStorage id="vtkMRMLSegmentationStorageNode_LNs" fileName="Data/All_Lymph_Node_Areas.seg.nrrd" />

  <SegmentationDisplay id="vtkMRMLSegmentationDisplayNode_Rest" opacity="0.25" visibility="false" visibility2D="true" visibility3D="true" Visibility2DFill="true" Visibility2DOutline="true" />
  <Segmentation id="vtkMRMLSegmentationNode_Rest" name="All_Rest" storageNodeRef="vtkMRMLSegmentationStorageNode_Rest" displayNodeRef="vtkMRMLSegmentationDisplayNode_Rest" referenceImageGeometryReference="vtkMRMLScalarVolumeNode1" />
  <SegmentationStorage id="vtkMRMLSegmentationStorageNode_Rest" fileName="Data/All_Rest.seg.nrrd" />

  <SegmentationDisplay id="vtkMRMLSegmentationDisplayNode_Helpers" opacity="0.2" visibility="false" visibility2D="true" visibility3D="true" Visibility2DFill="true" Visibility2DOutline="true" />
  <Segmentation id="vtkMRMLSegmentationNode_Helpers" name="All_Helpers" storageNodeRef="vtkMRMLSegmentationStorageNode_Helpers" displayNodeRef="vtkMRMLSegmentationDisplayNode_Helpers" referenceImageGeometryReference="vtkMRMLScalarVolumeNode1" />
  <SegmentationStorage id="vtkMRMLSegmentationStorageNode_Helpers" fileName="Data/All_Helpers.seg.nrrd" />

  <SegmentationDisplay id="vtkMRMLSegmentationDisplayNode_Vessels" opacity="0.4" visibility="false" visibility2D="true" visibility3D="true" Visibility2DFill="true" Visibility2DOutline="true" />
  
  <Segmentation id="vtkMRMLSegmentationNode_Abd" name="Abdominal_Organs" storageNodeRef="vtkMRMLSegmentationStorageNode4" displayNodeRef="vtkMRMLSegmentationDisplayNode_Abd"/>
  <SegmentationStorage id="vtkMRMLSegmentationStorageNode4" fileName="Data/Abdominal_Organs.seg.nrrd" />
  <Segmentation id="vtkMRMLSegmentationNode_NH" name="new_helpers" storageNodeRef="vtkMRMLSegmentationStorageNode5" displayNodeRef="vtkMRMLSegmentationDisplayNode_NH"/>
  <SegmentationStorage id="vtkMRMLSegmentationStorageNode5" fileName="Data/new_helpers.seg.nrrd" />
  <Segmentation id="vtkMRMLSegmentationNode_Vessels" name="Vessels" storageNodeRef="vtkMRMLSegmentationStorageNode_Vessels" displayNodeRef="vtkMRMLSegmentationDisplayNode_Vessels" referenceImageGeometryReference="vtkMRMLScalarVolumeNode1" />
  <SegmentationStorage id="vtkMRMLSegmentationStorageNode_Vessels" fileName="Data/Vessels.seg.nrrd" />

  <SegmentationDisplay id="vtkMRMLSegmentationDisplayNode_HelperPlanes" opacity="0.4" visibility="false" visibility2D="true" visibility3D="true" Visibility2DFill="true" Visibility2DOutline="true" />
  <Segmentation id="vtkMRMLSegmentationNode_HelperPlanes" name="helper_planes" storageNodeRef="vtkMRMLSegmentationStorageNode_HelperPlanes" displayNodeRef="vtkMRMLSegmentationDisplayNode_HelperPlanes" referenceImageGeometryReference="vtkMRMLScalarVolumeNode1" />
  <SegmentationStorage id="vtkMRMLSegmentationStorageNode_HelperPlanes" fileName="Data/helper_planes.seg.nrrd" />
'''

        for ext_model in external_model_nodes:
            new_mrml_str = f'''
  <SegmentationDisplay id="vtkMRMLSegmentationDisplayNode_{ext_model}" opacity="0.3" visibility="false" />
  <Segmentation id="vtkMRMLSegmentationNode_{ext_model}" name="{ext_model}" storageNodeRef="vtkMRMLSegmentationStorageNode_{ext_model}" displayNodeRef="vtkMRMLSegmentationDisplayNode_{ext_model}" referenceImageGeometryReference="vtkMRMLScalarVolumeNode1" />
  <SegmentationStorage id="vtkMRMLSegmentationStorageNode_{ext_model}" fileName="Data/{ext_model}.seg.nrrd" />
'''
            mrml_content += new_mrml_str
            
        mrml_content += "\n</MRML>\n"



        os.makedirs(os.path.dirname(os.path.abspath(output_mrb_path)), exist_ok=True)

        mrml_file = os.path.join(bundle_dir, f"{bundle_name}.mrml")
        with open(mrml_file, "w") as mf:
            mf.write(mrml_content)

        ct_file = os.path.join(data_dir, "CT.nrrd")
        nrrd.write(ct_file, ct_arr, nrrd_header_base, index_order='F')

        ln_file = os.path.join(data_dir, "All_Lymph_Node_Areas.seg.nrrd")
        nrrd.write(ln_file, np.transpose(ln_arr, (2, 1, 0)), ln_header, index_order='F')

        rest_file = os.path.join(data_dir, "All_Rest.seg.nrrd")
        nrrd.write(rest_file, rest_arr, rest_header, index_order='F')


        helper_file = os.path.join(data_dir, "All_Helpers.seg.nrrd")
        nrrd.write(helper_file, np.transpose(helper_arr, (2, 1, 0)), helper_header, index_order='F')

        vessels_file = os.path.join(data_dir, "Vessels.seg.nrrd")
        nrrd.write(vessels_file, np.transpose(vessels_arr, (2, 1, 0)), vessels_header, index_order='F')

        # Abdominal Organs into Abdominal_Organs.seg.nrrd
        abd_arr = np.zeros((dims[2], dims[1], dims[0]), dtype=np.uint16)
        abd_header = dict(nrrd_header_base)
        abd_header['type'] = 'unsigned short'
        abd_header['Segmentation_MasterRepresentation'] = 'Binary labelmap'
        abd_counter = 1
        abd_seg_count = 0
        
        target_abd = ["spleen", "kidney_left", "kidney_right", "liver", "gallbladder", "stomach", "helper_stomach_cardia", "helper_pancreas_head", "helper_pancreas_tail", "helper_pyloric_derived", "colon", "diaphragm_proxy", "small_bowel", "duodenum", "adrenal_gland_left", "adrenal_gland_right", "urinary_bladder", "rectum", "prostate", "uterus", "vagina", "seminal_vesicle", "esophagus", "liver_segment_1", "liver_segment_2", "liver_segment_3", "liver_segment_4", "liver_segment_5", "liver_segment_6", "liver_segment_7", "liver_segment_8"]
        for k in target_abd:
            k_lower = k.lower()
            m_raw = None
            if k_lower in res_keys_lower:
                m_raw = f_res[res_keys_lower[k_lower]][:]
            elif "masks" in f_prim and k_lower in [mk.lower() for mk in f_prim["masks"].keys()]:
                prim_keys_lower = {mk.lower(): mk for mk in f_prim["masks"].keys()}
                m_raw = f_prim["masks"][prim_keys_lower[k_lower]][:]
            
            if m_raw is not None:
                m_arr = m_raw
                fg = m_arr > 0
                if fg.sum() > 0:
                    abd_arr[fg] = abd_counter
                    tag = f"Segment{abd_seg_count}"
                    abd_header[f"{tag}_ID"] = f"Segment_{abd_counter}"
                    abd_header[f"{tag}_Name"] = k
                    abd_header[f"{tag}_LabelValue"] = str(abd_counter)
                    abd_header[f"{tag}_Layer"] = "0"
                    abd_header[f"{tag}_Color"] = get_color(k)
                    abd_counter += 1
                    abd_seg_count += 1
            elif "masks" in f_prim and k_lower in [mk.lower() for mk in f_prim["masks"].keys()]:
                prim_keys_lower = {mk.lower(): mk for mk in f_prim["masks"].keys()}
                m_raw = f_prim["masks"][prim_keys_lower[k_lower]][:]
                m_arr = m_raw
                fg = m_arr > 0
                if fg.sum() > 0:
                    abd_arr[fg] = abd_counter
                    tag = f"Segment{abd_seg_count}"
                    abd_header[f"{tag}_ID"] = f"Segment_{abd_counter}"
                    abd_header[f"{tag}_Name"] = k
                    abd_header[f"{tag}_LabelValue"] = str(abd_counter)
                    abd_header[f"{tag}_Layer"] = "0"
                    abd_header[f"{tag}_Color"] = get_color(k)
                    abd_counter += 1
                    abd_seg_count += 1
                    
        abd_file = os.path.join(data_dir, "Abdominal_Organs.seg.nrrd")
        nrrd.write(abd_file, np.transpose(abd_arr, (2, 1, 0)), abd_header, index_order='F')

        
        
        # New Helpers (Layered to support overlap)
        layers_nh = [np.zeros((dims[2], dims[1], dims[0]), dtype=np.uint16)]
        new_helpers_header = dict(nrrd_header_base)
        new_helpers_header['type'] = 'unsigned short'
        new_helpers_header['Segmentation_MasterRepresentation'] = 'Binary labelmap'
        nh_counter = 1
        nh_seg_count = 0
        
        target_nh = [
            # Computed helpers
            "helper_hilar_base_left",
            "helper_hilar_base_right",
            "helper_lung_arteries_left_side",
            "helper_lung_veins_left_side",
            "helper_lung_arteries_right_side",
            "helper_lung_veins_right_side",
            "helper_heart_chambers_hull",
            "helper_heart_chambers_costal_hull",
            "helper_pleural_space_left",
            "helper_pleural_space_right",
            "helper_vertebrae_dilated",
            "helper_prevascular_paratracheal_dilated",
            "helper_sacrum_fused",
            "helper_Abdominal_Presacral_Base",
            "helper_rectum_dilated",
            "helper_acetabulum_right",
            "helper_acetabulum_left",
            # TS masks used as hilar exclusions (from primary_masks.h5)
            "lung_upper_lobe_left", "lung_lower_lobe_left",
            "lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right",
            "lung_left", "lung_right",
            "heart",
            "aorta", "pulmonary_artery", "pulmonary_vein",
            "vertebrae", "spinal_cord",
            "trachea", "esophagus",
            "inferior_vena_cava", "superior_vena_cava",
            "costal_cartilages",
            "lung_arteries", "lung_veins"
        ]
        for k in target_nh:
            k_lower = k.lower()
            m_raw = None
            if k_lower in res_keys_lower:
                m_raw = f_res[res_keys_lower[k_lower]][:]
            elif "masks" in f_prim and k_lower in [mk.lower() for mk in f_prim["masks"].keys()]:
                prim_keys_lower = {mk.lower(): mk for mk in f_prim["masks"].keys()}
                m_raw = f_prim["masks"][prim_keys_lower[k_lower]][:]
                
            if m_raw is not None:
                m_arr = m_raw
                fg = m_arr > 0
                if fg.sum() > 0:
                    assigned_layer = -1
                    idx_1d = np.flatnonzero(fg)
                    if len(idx_1d) > 0:
                        if 'nh_bitmask' not in locals():
                            nh_bitmask = np.zeros(dims[0]*dims[1]*dims[2], dtype=np.uint64)
                        
                        used_bits = np.bitwise_or.reduce(nh_bitmask[idx_1d])
                        for i in range(60):
                            if not (used_bits & (np.uint64(1) << np.uint64(i))):
                                assigned_layer = i
                                break
                                
                        nh_bitmask[idx_1d] |= (np.uint64(1) << np.uint64(assigned_layer))
                        
                    if assigned_layer >= len(layers_nh):
                        for _ in range(assigned_layer - len(layers_nh) + 1):
                            layers_nh.append(np.zeros((dims[2], dims[1], dims[0]), dtype=np.uint16))
                            
                    layers_nh[assigned_layer].ravel()[idx_1d] = nh_counter
                    tag = f"Segment{nh_seg_count}"
                    new_helpers_header[f"{tag}_ID"] = f"Segment_{nh_counter}"
                    new_helpers_header[f"{tag}_Name"] = k
                    new_helpers_header[f"{tag}_LabelValue"] = str(nh_counter)
                    new_helpers_header[f"{tag}_Layer"] = str(assigned_layer)
                    new_helpers_header[f"{tag}_Color"] = get_color(k)
                    nh_counter += 1
                    nh_seg_count += 1

        if len(layers_nh) > 1:
            new_helpers_arr = np.transpose(np.stack(layers_nh, axis=-1), (2, 1, 0, 3))
            new_helpers_header['dimension'] = 4
            new_helpers_header['kinds'] = ['domain', 'domain', 'domain', 'list']
            sd = list(new_helpers_header['space directions'])
            sd.append([np.nan, np.nan, np.nan])
            new_helpers_header['space directions'] = np.array(sd)
            if 'sizes' in new_helpers_header: del new_helpers_header['sizes']
        else:
            new_helpers_arr = np.transpose(layers_nh[0], (2, 1, 0))

        nh_file = os.path.join(data_dir, "new_helpers.seg.nrrd")
        nrrd.write(nh_file, new_helpers_arr, new_helpers_header, index_order='F')




        helper_planes_file = os.path.join(data_dir, "helper_planes.seg.nrrd")
        nrrd.write(helper_planes_file, np.transpose(helper_planes_arr, (2, 1, 0)), helper_planes_header, index_order='F')

        print(f"  Zipping MRB bundle into: {output_mrb_path}...")
        with zipfile.ZipFile(output_mrb_path, "w", zipfile.ZIP_STORED) as z:
            for root, dirs, files in os.walk(bundle_dir):
                for f in files:
                    full_p = os.path.join(root, f)
                    rel_p = os.path.relpath(full_p, tmpdir)
                    z.write(full_p, rel_p)

        print(f"MRB generation successful: {output_mrb_path} ({os.path.getsize(output_mrb_path) / (1024*1024):.2f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean Slicer MRB Packager for Julia GPU Lymph Nodes")
    parser.add_argument("--h5", required=True, help="Path to primary_masks.h5")
    parser.add_argument("--results", required=True, help="Path to final_results_gpu.h5")
    parser.add_argument("--output-mrb", required=True, help="Output .mrb path")
    args = parser.parse_args()

    build_mrb(args.h5, args.results, args.output_mrb)
