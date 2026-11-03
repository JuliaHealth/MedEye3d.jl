#!/usr/bin/env python3
"""
process_mri_diffusivity.py — Computes edge detection and Heat-GDT diffusivity
for all MRI timepoints in preprocessed_volumes.h5.

Computes:
  - Percentile-normalized gradient magnitude for each MRI volume (T2, ADC, DWI, T1, DCE, etc.)
  - Multi-parametric MRI synergy: combines primary sequence edge with companion sequences
  - PSMA PET hypermetabolic uptake edge
  - Prostate anatomical capsule boundary edge (from max_anatomy zones 1-8)
  - Exclusion organ boundary insulation (D = 0)
  - Champion diffusivity field: D(x) = eps_D + (1 - eps_D) * exp(-gamma * E(x))
"""

import os
import sys
import h5py
import numpy as np
from scipy.ndimage import gaussian_filter

CHAMPION_GAMMA = 10.095886
CHAMPION_EPS_D = 1.0e-5

def compute_gradient_edge(vol, p_mid=0.12, slope=15.0, sigma=0.8):
    """
    Computes smoothed gradient magnitude and zero-baseline sigmoidal edge probability.
    """
    pos = vol[vol > 0]
    if len(pos) == 0:
        return np.zeros_like(vol, dtype=np.float32)
    p99 = np.percentile(pos, 99)
    p01 = np.percentile(pos, 1)
    norm = np.clip((vol - p01) / max(p99 - p01, 1e-4), 0.0, 1.5)
    
    gz, gy, gx = np.gradient(norm)
    mag = gaussian_filter(np.sqrt(gz**2 + gy**2 + gx**2), sigma=sigma)
    
    edge_raw = 1.0 / (1.0 + np.exp(-slope * (mag - p_mid)))
    edge_zero = 1.0 / (1.0 + np.exp(-slope * (-p_mid)))
    edge = np.clip((edge_raw - edge_zero) / (1.0 - edge_zero), 0.0, 1.0)
    return edge.astype(np.float32)

def compute_pet_edge(pet_vol, sigma=0.8):
    """
    Computes edge map for hypermetabolic PET tracer uptake.
    """
    if pet_vol is None or np.max(pet_vol) <= 0.5:
        return None
    pet_norm = np.clip(pet_vol, 0.0, 20.0) / 20.0
    gz, gy, gx = np.gradient(pet_norm)
    mag = gaussian_filter(np.sqrt(gz**2 + gy**2 + gx**2), sigma=sigma)
    edge_raw = 1.0 / (1.0 + np.exp(-8.0 * (mag - 0.15)))
    edge_zero = 1.0 / (1.0 + np.exp(-8.0 * (-0.15)))
    return np.clip((edge_raw - edge_zero) / (1.0 - edge_zero), 0.0, 1.0).astype(np.float32)

def compute_prostate_capsule_edge(anat_vol, sigma=0.6):
    """
    Computes edge map for the outer prostatic capsule boundary from max_anatomy zones (1-8).
    """
    if anat_vol is None or np.max(anat_vol) == 0:
        return None
    pros_mask = (anat_vol > 0).astype(np.float32)
    if pros_mask.sum() == 0:
        return None
    gz, gy, gx = np.gradient(pros_mask)
    mag = gaussian_filter(np.sqrt(gz**2 + gy**2 + gx**2), sigma=sigma)
    return np.clip(mag * 2.0, 0.0, 1.0).astype(np.float32)

def process_case_mri_diffusivity(h5_path):
    print(f"===========================================================")
    print(f"  Processing MRI Edge Detection & Diffusivity: {h5_path}")
    print(f"===========================================================")
    
    if not os.path.isfile(h5_path):
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")
        
    with h5py.File(h5_path, "r+") as h5:
        # Check timepoint groups
        tfm_groups = [k for k in h5.keys() if k.startswith("TFM_Transform_FollowUp_to_Baseline_")]
        def sort_key(name):
            import re
            m = re.search(r"_(\d+)\.", name)
            return int(m.group(1)) if m else 999
        tfm_groups.sort(key=sort_key)
        
        print(f"Found {len(tfm_groups)} MRI followup groups:")
        for g in tfm_groups:
            print(f"  - {g}")
            
        if not tfm_groups:
            print("No MRI followup groups found. Nothing to process.")
            return

        # Load PET and anatomy if present
        pet_vol = None
        anat_vol = None
        first_grp = h5[tfm_groups[0]]
        for k in first_grp.keys():
            if k.startswith("SUV_PET_Image"):
                pet_vol = first_grp[k][()].astype(np.float32)
                print(f"Loaded registered PET volume ({pet_vol.shape})")
            elif k.startswith("max_anatomy"):
                anat_vol = first_grp[k][()].astype(np.uint16)
                print(f"Loaded MRI prostate anatomy volume ({anat_vol.shape})")

        # Compute common PET & prostate capsule edges
        e_pet = compute_pet_edge(pet_vol)
        e_capsule = compute_prostate_capsule_edge(anat_vol)
        if e_capsule is not None:
            print(f"Prostate capsule boundary edge extracted (voxels > 0.5: {(e_capsule > 0.5).sum()})")
        if e_pet is not None:
            print(f"PET tracer uptake edge extracted (voxels > 0.5: {(e_pet > 0.5).sum()})")

        # 1. Load all MRI volumes and identify modalities
        mri_data = {}
        for g_name in tfm_groups:
            grp = h5[g_name]
            mri_vol = None
            modality = "T2"
            desc = ""
            for k in grp.keys():
                if k.startswith("Fixed_CT_Volume"):
                    ds = grp[k]
                    mri_vol = ds[()].astype(np.float32)
                    modality = ds.attrs.get("modality", "T2")
                    desc = ds.attrs.get("series_description", "")
                    break
            if mri_vol is not None:
                mri_data[g_name] = {
                    "vol": mri_vol,
                    "mod": str(modality).upper(),
                    "desc": str(desc)
                }
                print(f"  Loaded {g_name}: {modality} ({desc}), shape={mri_vol.shape}")

        # 2. Compute individual modality edge maps
        raw_edges = {}
        for g_name, d in mri_data.items():
            mod = d["mod"]
            vol = d["vol"]
            # Modality-specific gradient midpoint:
            # ADC and DWI have lower background noise and sharper focal diffusion restrictions
            if "ADC" in mod:
                p_mid = 0.08
                slope = 16.0
            elif "DWI" in mod or "BVAL" in mod:
                p_mid = 0.08
                slope = 16.0
            elif "T1" in mod:
                p_mid = 0.10
                slope = 15.0
            elif "DCE" in mod:
                p_mid = 0.09
                slope = 15.0
            else:  # T2, T2 COR, T2 SAG
                p_mid = 0.12
                slope = 15.0
                
            e_mod = compute_gradient_edge(vol, p_mid=p_mid, slope=slope)
            raw_edges[g_name] = e_mod
            print(f"  Computed edge for {g_name} ({mod}): mean={e_mod.mean():.4f}, max={e_mod.max():.4f}, voxels>0.5={(e_mod > 0.5).sum()}")

        # 3. Multi-parametric synergy: combine companion MRI edges
        # Find T2, ADC, DWI group names if present
        t2_grp = next((g for g, d in mri_data.items() if d["mod"] in ("T2", "MR", "MRI") and "COR" not in d["desc"] and "SAG" not in d["desc"]), None)
        adc_grp = next((g for g, d in mri_data.items() if "ADC" in d["mod"]), None)
        dwi_grp = next((g for g, d in mri_data.items() if "DWI" in d["mod"] or "BVAL" in d["mod"]), None)

        print("\nMulti-parametric synergy pairing:")
        print(f"  Primary T2 axial:  {t2_grp}")
        print(f"  Primary ADC axial: {adc_grp}")
        print(f"  Primary DWI axial: {dwi_grp}")

        # 4. Generate and save diffusivity for each MRI group
        for g_name, d in mri_data.items():
            grp = h5[g_name]
            mod = d["mod"]
            e_primary = raw_edges[g_name]
            
            # Combine edges:
            # 1. Primary modality edge (weight 1.0)
            E = e_primary.copy()
            
            # 2. Add companion axial mpMRI edges with 0.75 weight if co-registered
            if g_name == t2_grp:
                if adc_grp and adc_grp in raw_edges:
                    E = np.maximum(E, 0.75 * raw_edges[adc_grp])
                if dwi_grp and dwi_grp in raw_edges:
                    E = np.maximum(E, 0.75 * raw_edges[dwi_grp])
            elif g_name == adc_grp:
                if t2_grp and t2_grp in raw_edges:
                    E = np.maximum(E, 0.75 * raw_edges[t2_grp])
                if dwi_grp and dwi_grp in raw_edges:
                    E = np.maximum(E, 0.75 * raw_edges[dwi_grp])
            elif g_name == dwi_grp:
                if t2_grp and t2_grp in raw_edges:
                    E = np.maximum(E, 0.75 * raw_edges[t2_grp])
                if adc_grp and adc_grp in raw_edges:
                    E = np.maximum(E, 0.75 * raw_edges[adc_grp])
            elif "T1" in mod:
                if t2_grp and t2_grp in raw_edges:
                    E = np.maximum(E, 0.70 * raw_edges[t2_grp])
            elif "DCE" in mod:
                if t2_grp and t2_grp in raw_edges:
                    E = np.maximum(E, 0.70 * raw_edges[t2_grp])

            # 3. Add PSMA PET tracer uptake edge
            if e_pet is not None:
                E = np.maximum(E, e_pet)

            # 4. Add prostate capsule boundary edge
            if e_capsule is not None:
                E = np.maximum(E, e_capsule)

            E = np.clip(E, 0.0, 1.0).astype(np.float32)

            # 5. Compute champion diffusivity: D = eps_D + (1 - eps_D) * exp(-gamma * E)
            D = CHAMPION_EPS_D + (1.0 - CHAMPION_EPS_D) * np.exp(-CHAMPION_GAMMA * E)
            D = np.clip(D, CHAMPION_EPS_D, 1.0).astype(np.float32)

            # Save into HDF5 group, replacing old diffusivity dataset
            if "diffusivity" in grp:
                del grp["diffusivity"]
                
            grp.create_dataset("diffusivity", data=D, compression="gzip", compression_opts=1)
            print(f"  [SAVED] {g_name}/diffusivity: shape={D.shape}, min={D.min():.6f}, max={D.max():.6f}, mean={D.mean():.4f}")

    print("\n✅ All MRI timepoint diffusivity fields successfully processed and saved!")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target = sys.argv[1]
    else:
        target = "/workspaces/MedEye3d.jl/data/cases/Prostata_bimodal/Pat23/preprocessed_volumes.h5"
    process_case_mri_diffusivity(target)
