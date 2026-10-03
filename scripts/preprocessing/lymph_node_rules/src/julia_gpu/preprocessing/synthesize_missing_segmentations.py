#!/usr/bin/env python3
"""
Synthesize missing segmentations from available TotalSegmentator and external model outputs.

This script bridges the gap between TS 2.18 output naming and the lymph node pipeline's
expected input segmentations. It creates:
  - Combined lung masks from individual lobes (TS 2.18 change)
  - lung_trachea_bronchia from trachea + lung_airways (old TS class removed)
  - Aliases for renamed structures (coracobrachial→coracobrachialis, etc.)
  - rectum from inferior portion of colon (when NV-Segment unavailable)
  - MuscleMap output splitting (multi-label → individual NIfTI files)

Run AFTER TotalSegmentator and external models, BEFORE Julia preprocessing.
"""
import os
import sys
import numpy as np
import nibabel as nib

# MuscleMap label → output filename mapping
MUSCLEMAP_LABELS = {
    1101: "levator_scapulae_left",
    1102: "levator_scapulae_right",
    1141: "sternocleidomastoid_left",
    1142: "sternocleidomastoid_right",
    1151: "longus_colli_left",
    1152: "longus_colli_right",
    1161: "trapezius_left",
    1162: "trapezius_right",
    2101: "supraspinatus_left",
    2102: "supraspinatus_right",
    2111: "subscapularis_left",
    2112: "subscapularis_right",
    2121: "infraspinatus_left",
    2122: "infraspinatus_right",
    2141: "deltoid_left",
    2142: "deltoid_right",
    5101: "transversospinalis_left",
    5102: "transversospinalis_right",
    5111: "erector_spinae_left",
    5112: "erector_spinae_right",
    5121: "psoas_major_left",
    5122: "psoas_major_right",
    5131: "quadratus_lumborum_left",
    5132: "quadratus_lumborum_right",
    5141: "latissimus_dorsi_left",
    5142: "latissimus_dorsi_right",
    6101: "gluteus_minimus_left",
    6102: "gluteus_minimus_right",
    6111: "gluteus_medius_left",
    6112: "gluteus_medius_right",
    6121: "gluteus_maximus_left",
    6122: "gluteus_maximus_right",
    6141: "iliacus_left",
    6142: "iliacus_right",
    6171: "femur_left",
    6172: "femur_right",
    6181: "piriformis_left",
    6182: "piriformis_right",
    6201: "obturator_internus_left",
    6202: "obturator_internus_right",
    6211: "obturator_externus_left",
    6212: "obturator_externus_right",
}


def synthesize(seg_dir, ct_path=None):
    """Create missing segmentations from available ones."""

    def has(name):
        return os.path.exists(os.path.join(seg_dir, f"{name}.nii.gz"))

    def load(name):
        p = os.path.join(seg_dir, f"{name}.nii.gz")
        return nib.load(p) if os.path.exists(p) else None

    def save(name, arr, ref, force=False):
        p = os.path.join(seg_dir, f"{name}.nii.gz")
        if os.path.exists(p) and not force:
            return  # don't overwrite existing
        nib.save(nib.Nifti1Image(arr.astype(np.uint8), ref.affine, ref.header), p)
        print(f"  [Synth] {name}: {np.count_nonzero(arr)} voxels")

    # Get a reference image for shape/affine
    ref = None
    if ct_path and os.path.exists(ct_path):
        ref = nib.load(ct_path)
    else:
        for f in os.listdir(seg_dir):
            if f.endswith('.nii.gz'):
                ref = nib.load(os.path.join(seg_dir, f))
                break

    if ref is None:
        print("  [Synth] ERROR: No reference image found")
        return

    # ==========================================
    # 1. Combined lungs from individual lobes
    # ==========================================
    for side, lobes in [("left", ["lung_upper_lobe_left", "lung_lower_lobe_left"]),
                        ("right", ["lung_upper_lobe_right", "lung_middle_lobe_right",
                                   "lung_lower_lobe_right"])]:
        # Always overwrite lung_left/lung_right with combined lobes:
        # TS 'total' task produces tiny hilum-only lung_left/lung_right (~15-19k voxels)
        # which are useless for exclusion. The lobe-combined version (~400-600k voxels)
        # is always the correct mask to use.
        c = np.zeros(ref.shape, dtype=np.uint8)
        found = False
        for l in lobes:
            img = load(l)
            if img is not None:
                c |= (img.get_fdata() > 0).astype(np.uint8)
                found = True
        if found:
            save(f"lung_{side}", c, ref, force=True)

    # ==========================================
    # 2. lung_trachea_bronchia = trachea + lung_airways
    # ==========================================
    if not has("lung_trachea_bronchia"):
        c = np.zeros(ref.shape, dtype=np.uint8)
        found = False
        for src in ["trachea", "lung_airways"]:
            img = load(src)
            if img is not None:
                c |= (img.get_fdata() > 0).astype(np.uint8)
                found = True
        if found:
            save("lung_trachea_bronchia", c, ref)

    # ==========================================
    # 3. Aliases for renamed TS classes
    # ==========================================
    aliases = [
        ("coracobrachial", "coracobrachialis"),
        ("larynx_air", "larynx"),
    ]
    for old, new in aliases:
        if not has(new) and has(old):
            img = load(old)
            save(new, (img.get_fdata() > 0).astype(np.uint8), img)

    # ==========================================
    # 4. rectum from inferior portion of colon (fallback)
    # ==========================================
    if not has("rectum") and has("colon"):
        c_img = load("colon")
        c = c_img.get_fdata()
        zi = np.where(c.max(axis=(0, 1)) > 0)[0]
        if len(zi) > 5:
            z_range = zi.max() - zi.min()
            z_cut = zi.min() + z_range // 5  # inferior 20%
            r = np.zeros_like(c, dtype=np.uint8)
            r[:, :, :z_cut + 1] = (c[:, :, :z_cut + 1] > 0).astype(np.uint8)
            if np.count_nonzero(r) > 100:
                save("rectum", r, c_img)

    # ==========================================
    # 5. Split MuscleMap multi-label output into individual files
    # ==========================================
    mm_files = [f for f in os.listdir(seg_dir) if "dseg" in f.lower() and f.endswith(".nii.gz")]
    for mm_file in mm_files:
        mm_path = os.path.join(seg_dir, mm_file)
        mm_img = nib.load(mm_path)
        mm_data = mm_img.get_fdata().astype(np.int32)
        unique_labels = np.unique(mm_data)
        unique_labels = unique_labels[unique_labels > 0]
        print(f"  [Synth] Processing MuscleMap output {mm_file}: {len(unique_labels)} labels found")
        for label in unique_labels:
            label_int = int(label)
            if label_int in MUSCLEMAP_LABELS:
                name = MUSCLEMAP_LABELS[label_int]
                if not has(name):
                    mask = (mm_data == label).astype(np.uint8)
                    save(name, mask, mm_img)

    # ==========================================
    # 6. splenic_vein from portal_vein_and_splenic_vein
    # ==========================================
    if not has("splenic_vein") and has("portal_vein_and_splenic_vein"):
        img = load("portal_vein_and_splenic_vein")
        save("splenic_vein", (img.get_fdata() > 0).astype(np.uint8), img)

    # ==========================================
    # 7. Split iliac vessels into common/external at L5 bottom / aorta bottom
    # ==========================================
    # The pipeline needs iliac_artery_common_left/right and iliac_artery_external_left/right
    # but TotalSegmentator and NV-Segment only produce undivided iliac_artery_left/right.
    # Anatomy:
    #   - Common iliac = short segment from aortic bifurcation (aorta bottom) down to
    #     iliac bifurcation (~L5/S1 level)
    #   - External iliac = from iliac bifurcation down to inguinal ligament
    # Higher slice index = more superior in this coordinate system.
    l5_path = os.path.join(seg_dir, "vertebrae_L5.nii.gz")
    aorta_path = os.path.join(seg_dir, "aorta.nii.gz")
    if os.path.exists(l5_path):
        l5_data = nib.load(l5_path).get_fdata().astype(np.uint8)
        l5_slices = np.where(l5_data.sum(axis=(0, 1)) > 0)[0]
        # Find aorta bottom as upper cap for common iliac
        aorta_cap = None
        if os.path.exists(aorta_path):
            aorta_data = nib.load(aorta_path).get_fdata().astype(np.uint8)
            aorta_slices = np.where(aorta_data.sum(axis=(0, 1)) > 0)[0]
            if len(aorta_slices) > 0:
                aorta_cap = int(aorta_slices.min())  # Aortic bifurcation = bottom of aorta
        if len(l5_slices) > 0:
            l5_bottom = int(l5_slices.min())  # Bottom of L5 = iliac bifurcation level
            for vtype in ["artery", "vena"]:
                for side in ["left", "right"]:
                    whole_name = f"iliac_{vtype}_{side}"
                    common_name = f"iliac_{vtype}_common_{side}"
                    external_name = f"iliac_{vtype}_external_{side}"
                    if has(whole_name) and not (has(common_name) and has(external_name)):
                        whole_img = load(whole_name)
                        whole_data = whole_img.get_fdata().astype(np.uint8)
                        # Common = between L5 bottom and aorta bottom (near bifurcation)
                        common = whole_data.copy()
                        common[:, :, :l5_bottom] = 0          # Remove below L5 (external territory)
                        if aorta_cap is not None:
                            common[:, :, aorta_cap:] = 0      # Remove above aorta bottom (aorta territory)
                        # External = below L5 bottom (going to inguinal)
                        external = whole_data.copy()
                        external[:, :, l5_bottom:] = 0        # Remove above L5 (common territory)
                        c_count = int(np.sum(common > 0))
                        e_count = int(np.sum(external > 0))
                        if c_count > 0:
                            save(common_name, common, whole_img)
                            print(f"  [Synth] Split {whole_name} → {common_name}: {c_count} voxels (slices {l5_bottom}-{aorta_cap if aorta_cap else '?'})")
                        if e_count > 0:
                            save(external_name, external, whole_img)
                            print(f"  [Synth] Split {whole_name} → {external_name}: {e_count} voxels (slices <{l5_bottom})")
    # ==========================================
    # 8. Split aorta into ascending/descending below the aortic arch
    # ==========================================
    # The aorta below the arch has 2 connected components per slice:
    # anterior (lower Y) = ascending, posterior (higher Y) = descending.
    # These are needed by LineLimitBetweenLandmarks in subaortic/paratracheal rules.
    if has("aorta") and (not has("aorta_ascending") or not has("aorta_descending")):
        from scipy import ndimage
        aorta_img = load("aorta")
        aorta_data = (aorta_img.get_fdata() > 0).astype(np.uint8)
        asc = np.zeros_like(aorta_data)
        desc = np.zeros_like(aorta_data)

        # Find the arch: highest slice range where aorta has only 1 component
        # Below arch: 2 components (ascending + descending)
        arch_top = 0
        for z in range(aorta_data.shape[2] - 1, -1, -1):
            slc = aorta_data[:, :, z]
            if slc.sum() == 0:
                continue
            labeled, n = ndimage.label(slc)
            if n >= 2:
                arch_top = z + 1  # first slice above the split
                break

        # Split each slice below arch into ascending (anterior) and descending (posterior)
        for z in range(aorta_data.shape[2]):
            slc = aorta_data[:, :, z]
            if slc.sum() == 0:
                continue
            labeled, n = ndimage.label(slc)
            if n >= 2:
                # Find center of mass for each component
                # In our coord system, Y axis is anterior-posterior
                # Ascending aorta is anterior (lower Y values)
                centers = ndimage.center_of_mass(slc, labeled, range(1, n + 1))
                # Sort by Y coordinate (dim 1 in the slice)
                comp_y = [(i + 1, c[1]) for i, c in enumerate(centers)]
                comp_y.sort(key=lambda x: x[1])
                # Anterior (min Y) = ascending, posterior (max Y) = descending
                asc_label = comp_y[0][0]
                desc_label = comp_y[-1][0]
                asc[:, :, z] = (labeled == asc_label).astype(np.uint8)
                desc[:, :, z] = (labeled == desc_label).astype(np.uint8)
            elif z < arch_top:
                # Only 1 component below arch — assign based on last known split
                # or skip (rare edge case at transition)
                pass
            # Above arch: aorta is unified, no split needed

        asc_nz = int(np.count_nonzero(asc))
        desc_nz = int(np.count_nonzero(desc))
        if asc_nz > 0:
            save("aorta_ascending", asc, aorta_img, force=True)
        if desc_nz > 0:
            save("aorta_descending", desc, aorta_img, force=True)
    # ==========================================
    # 9. Synthesize ribs_combined and thorax_wall
    # ==========================================
    # ribs_combined = union of all rib_left_N + rib_right_N
    # thorax_wall = ribs_combined | sternum | costal_cartilages (+ morphological closing)
    # Required by Axillary_Level_II (Thoracic_Chest_Wall rule)
    if not has("ribs_combined"):
        rib_names = [f"rib_left_{i}" for i in range(1, 13)] + [f"rib_right_{i}" for i in range(1, 13)]
        present_ribs = [r for r in rib_names if has(r)]
        if len(present_ribs) >= 4:
            ref_img = load(present_ribs[0])
            combined = np.zeros(ref_img.get_fdata().shape, dtype=np.uint8)
            for rn in present_ribs:
                combined |= (load(rn).get_fdata() > 0).astype(np.uint8)
            rc_count = int(np.count_nonzero(combined))
            if rc_count > 0:
                save("ribs_combined", combined, ref_img, force=True)
                print(f"  [Synth] ribs_combined: {rc_count} voxels from {len(present_ribs)} ribs")

    if not has("thorax_wall"):
        from scipy import ndimage as ndi_tw
        parts = []
        ref_img_tw = None
        for part_name in ["ribs_combined", "sternum", "costal_cartilages"]:
            if has(part_name):
                img = load(part_name)
                if ref_img_tw is None:
                    ref_img_tw = img
                parts.append((img.get_fdata() > 0).astype(np.uint8))
        if len(parts) >= 2 and ref_img_tw is not None:
            wall = np.zeros(parts[0].shape, dtype=np.uint8)
            for p in parts:
                wall |= p
            # Morphological closing to fill intercostal gaps (3D ball radius ~3 voxels)
            struct = ndi_tw.generate_binary_structure(3, 1)  # 6-connectivity
            wall_closed = ndi_tw.binary_closing(wall, structure=struct, iterations=3).astype(np.uint8)
            tw_count = int(np.count_nonzero(wall_closed))
            if tw_count > 0:
                save("thorax_wall", wall_closed, ref_img_tw, force=True)
                print(f"  [Synth] thorax_wall: {tw_count} voxels (from {len(parts)} parts + morphological closing)")

    total = len([f for f in os.listdir(seg_dir) if f.endswith('.nii.gz')])
    print(f"  [Synth] Synthesis complete. Total segmentation files: {total}")


if __name__ == "__main__":
    case_dir = sys.argv[1]
    ct = os.path.join(case_dir, "Fixed_CT_Volume.nii.gz")
    seg = os.path.join(case_dir, "segmentations")
    if not os.path.exists(ct):
        ct = os.path.join(case_dir, "CT.nii.gz")
    synthesize(seg, ct)
