#!/usr/bin/env python3
"""
create_petct_student_dataset.py

Creates a PET/CT-only student dataset from Pat23's preprocessed_volumes.h5.
Two timepoints:
  TP0 (BASELINE)       : PET + CT volumes, blank lesion mask (for student practice)
  TP1 (Follow-up #1)   : PET + CT volumes, with expert lesion annotations (reference)

Also refreshes lymph node station labels in max_anatomy to ensure consistency
with current lymph node rules.

Usage (inside Docker container):
  python3 scripts/preprocessing/create_petct_student_dataset.py

Output:
  data/cases/Prostata_bimodal/Pat23_PETCT_student/preprocessed_volumes.h5
"""

import h5py
import json
import numpy as np
import os
import sys
from datetime import datetime


# ============================================================================
# Paths
# ============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SOURCE_H5 = os.path.join(BASE_DIR, "data", "cases", "Prostata_bimodal", "Pat23", "preprocessed_volumes.h5")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "cases", "Prostata_bimodal", "Pat23_PETCT_student")
OUTPUT_H5 = os.path.join(OUTPUT_DIR, "preprocessed_volumes.h5")
ONTOLOGY_PATH = os.path.join(BASE_DIR, "assets", "max_anatomy_to_ontology.json")
DATA_ONTOLOGY_PATH = os.path.join(BASE_DIR, "data", "max_anatomy_to_ontology.json")


def load_source_data(src_path):
    """Load all necessary data from the source HDF5 file."""
    print(f"[1/6] Loading source data from: {src_path}")
    f = h5py.File(src_path, "r")

    data = {}

    # --- BASELINE volumes (PET/CT) ---
    base = f["BASELINE"]
    data["ct"] = base["Fixed_CT_Volume_0.nii.gz"][:]
    data["pet"] = base["SUV_PET_Image_0.nii.gz"][:]
    data["mask_expert"] = base["PET_Lesions_0.nii.gz_expert"][:]
    data["mask_auto"] = base["PET_Lesions_0.nii.gz"][:]
    data["diffusivity"] = base["diffusivity"][:]
    data["max_anatomy"] = base["max_anatomy.nii.gz"][:]
    data["max_anatomy_expert"] = base["max_anatomy.nii.gz_expert"][:]
    data["skellytour"] = base["skellytour.nii.gz"][:]
    # Load dataset attributes (spacing, origin, direction)
    data["ct_attrs"] = dict(base["Fixed_CT_Volume_0.nii.gz"].attrs)
    data["pet_attrs"] = dict(base["SUV_PET_Image_0.nii.gz"].attrs)
    data["mask_attrs"] = dict(base["PET_Lesions_0.nii.gz"].attrs) if dict(base["PET_Lesions_0.nii.gz"].attrs) else {}
    print(f"  Loaded BASELINE: CT{data['ct'].shape}, PET{data['pet'].shape}, "
          f"mask{data['mask_expert'].shape}")
    print(f"  CT attrs: {list(data['ct_attrs'].keys())}")

    # --- ATLAS ---
    atlas = f["ATLAS"]
    data["atlas_bone"] = atlas["bone_atlas"][:]
    data["atlas_max_anatomy"] = atlas["max_anatomy"][:]
    data["atlas_max_anatomy_expert"] = atlas["max_anatomy_expert"][:]
    data["atlas_skellytour"] = atlas["skellytour"][:]
    print(f"  Loaded ATLAS: max_anatomy{data['atlas_max_anatomy'].shape}")

    # --- _meta_ ---
    meta = {}
    for key in f["_meta_"].keys():
        ds = f["_meta_"][key]
        val = ds[()]
        if isinstance(val, bytes):
            meta[key] = val.decode("utf-8")
        elif isinstance(val, np.ndarray):
            meta[key] = val
        elif isinstance(val, (np.integer, np.floating)):
            meta[key] = val
        else:
            meta[key] = val
    data["meta"] = meta
    print(f"  Loaded _meta_ keys: {list(meta.keys())}")

    # --- CENTROIDS for TP0 ---
    centroids = {}
    if "CENTROIDS" in f:
        for k in f["CENTROIDS"].keys():
            if k.startswith("tp0_"):
                centroids[k] = f["CENTROIDS"][k][:]
    data["centroids_tp0"] = centroids
    print(f"  Loaded {len(centroids)} TP0 centroids")

    # --- BONE_SUBSEG for TP0 ---
    bone_subsegs = {}
    if "BONE_SUBSEG" in f:
        for k in f["BONE_SUBSEG"].keys():
            if k.startswith("PET_Lesions_0_") or k.startswith("tp_0_"):
                bone_subsegs[k] = f["BONE_SUBSEG"][k][:]
    data["bone_subsegs_tp0"] = bone_subsegs
    print(f"  Loaded {len(bone_subsegs)} TP0 bone sub-segmentations")

    f.close()
    return data


def build_metadata_for_student(source_meta):
    """Build metadata.json for the student PET/CT dataset (2 TPs)."""
    src_metadata = json.loads(source_meta["metadata.json"]) if isinstance(source_meta["metadata.json"], str) else source_meta["metadata.json"]

    # First element is the patient info
    patient_info = dict(src_metadata[0])
    patient_info["Modalities"] = ["CT", "PET"]
    patient_info["CreatedDate"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    patient_info["PatientID"] = "Pat23_PETCT_student"

    # TP0 study info (blank — for practice)
    tp0_study = {
        "20230831_PET_Practice": {
            "Description": "PET/CT-Scan (Praxis-Datensatz, keine Annotationen)",
            "EnglishDescription": "PET/CT scan (practice dataset, no annotations). "
                                  "Students should annotate lesions in this timepoint.",
            "Modality": "PET",
            "PET": {"name": "SUV_PET_Image_0", "Modality": "PET", "StudyDate": "20230831"},
            "CT": {"name": "Fixed_CT_Volume_0", "Modality": "CT", "StudyDate": "20230831"}
        }
    }

    # TP1 study info (reference — with annotations)
    tp1_study = {
        "20230831_PET_Reference": {
            "Description": "PET/CT-Scan (Referenz-Datensatz mit Experten-Annotationen)",
            "EnglishDescription": "PET/CT scan (reference dataset with expert annotations). "
                                  "Compare your annotations against the expert reference.",
            "Modality": "PET",
            "PET": {"name": "SUV_PET_Image_1", "Modality": "PET", "StudyDate": "20230831"},
            "CT": {"name": "Fixed_CT_Volume_1", "Modality": "CT", "StudyDate": "20230831"}
        }
    }

    return [patient_info, tp0_study, tp1_study]


def build_scene_hierarchy():
    """Build scene_hierarchy.json for 2-TP PET/CT dataset."""
    hierarchy = [
        # TP0 — BASELINE (blank, for practice)
        {"name": "Fixed_CT_Volume_0", "type": "vtkMRMLScalarVolumeNode", "modality": "CT"},
        {"name": "SUV_PET_Image_0", "type": "vtkMRMLScalarVolumeNode", "modality": "PET"},
        {"name": "PET_Lesions_0", "type": "vtkMRMLSegmentationNode", "segments": []},
        {"name": "max_anatomy_fixed_ct_0", "source": "max_anatomy.nii.gz", "type": "vtkMRMLSegmentationNode"},
        {"name": "skellytour_fixed_ct_0", "source": "skellytour.nii.gz", "type": "vtkMRMLSegmentationNode"},
        # TP1 — Follow-up with expert annotations
        {
            "name": "Transform_FollowUp_to_Baseline_1",
            "type": "vtkMRMLLinearTransformNode",
            "transform_type": "Rigid/Affine (Linear)",
            "modality": "PET",
            "children": [
                {"name": "Fixed_CT_Volume_1", "type": "vtkMRMLScalarVolumeNode", "modality": "CT",
                 "series_description": "Reference CT (Expert Annotations)"},
                {"name": "SUV_PET_Image_1", "type": "vtkMRMLScalarVolumeNode", "modality": "PET"},
                {"name": "PET_Lesions_1", "type": "vtkMRMLSegmentationNode",
                 "segments": []},  # Will be populated below
                {"name": "max_anatomy_fixed_ct_1", "source": "max_anatomy.nii.gz",
                 "type": "vtkMRMLSegmentationNode"}
            ]
        }
    ]
    return hierarchy


def build_segment_names_for_tp1(source_meta):
    """Build segment names for TP1 (reference) from source TP0 segment names."""
    src_sn = json.loads(source_meta["segment_names.json"]) if isinstance(source_meta["segment_names.json"], str) else source_meta["segment_names.json"]
    # Source TP0 has the PET/CT segment names
    tp0_names = src_sn.get("0", {})
    return {
        "0": {},       # TP0 is blank
        "1": tp0_names  # TP1 has the expert annotations
    }


def build_organ_mapping_for_tp1(source_meta):
    """Build organ mapping for TP1 (reference) from source TP0."""
    src_om = json.loads(source_meta["organ_mapping"]) if isinstance(source_meta["organ_mapping"], str) else source_meta["organ_mapping"]
    tp0_mapping = src_om.get("0", {})
    return {
        "0": {},         # TP0 blank
        "1": tp0_mapping  # TP1 reference
    }


def build_lesion_annotations_for_tp1(source_meta):
    """Build lesion annotations for TP1 from source TP0."""
    src_la = json.loads(source_meta["lesion_annotations.json"]) if isinstance(source_meta["lesion_annotations.json"], str) else source_meta["lesion_annotations.json"]
    # Return all annotations (they reference TP0 in source = TP1 in student)
    return src_la


def build_matches_for_tp1(source_meta):
    """Build matches for TP1 from source — simplified for 2 TPs."""
    # Empty matches for student dataset (no cross-TP matching needed initially)
    return []


def filter_petct_lesion_ids(mask):
    """Get unique lesion IDs from the PET/CT mask (excluding 0)."""
    unique = np.unique(mask)
    return [int(u) for u in unique if u != 0]


def populate_scene_segments(hierarchy, mask_expert):
    """Populate segment names in scene_hierarchy from the mask."""
    lesion_ids = filter_petct_lesion_ids(mask_expert)
    # For TP1 reference — add segment names
    for item in hierarchy:
        if item.get("name") == "Transform_FollowUp_to_Baseline_1" and "children" in item:
            for child in item["children"]:
                if child.get("name") == "PET_Lesions_1":
                    child["segments"] = [f"Lesion {lid}" for lid in sorted(lesion_ids)]
    return hierarchy


def refresh_ontology_labels(ontology_path, max_anatomy_labels):
    """
    Ensure all labels in max_anatomy_labels.json have corresponding entries
    in the ontology file. Add missing lymph node station entries.
    Returns the (possibly updated) ontology dict.
    """
    print(f"[3/6] Refreshing ontology labels from: {ontology_path}")

    with open(ontology_path, "r") as f:
        ontology = json.load(f)

    # Build lookup for all anatomy label names
    label_names = set(max_anatomy_labels.values())

    missing = []
    for name in label_names:
        # Check both exact case and lowercase
        if name not in ontology and name.lower() not in ontology:
            missing.append(name)

    if missing:
        print(f"  Found {len(missing)} labels missing from ontology: {missing[:10]}...")
        for name in missing:
            # Determine the type of this label
            if any(kw in name.lower() for kw in [
                "iliac", "inguinal", "obturator", "mesenteric", "paraaortic",
                "presacral", "pararectal", "celiac", "paracardial", "station",
                "axillary", "rotter", "neck_level", "neck_parotid", "neck_retro",
                "thoracic_", "supraclavicular", "mammary", "prepericardial",
                "hilar", "subcarinial", "subaortic", "prevascular", "paratracheal",
                "retrotracheal", "paraoesophageal", "splenic", "renal_hilar"
            ]):
                # Lymph node station
                side = ""
                if "_left" in name.lower() or name.endswith("_Left"):
                    side = "Left"
                elif "_right" in name.lower() or name.endswith("_Right"):
                    side = "Right"

                # Determine sublocation
                subloc = "Abdominal / Mesenteric"
                if "thoracic" in name.lower() or "supraclavicular" in name.lower():
                    subloc = "Anterior Mediastinum"
                elif "neck" in name.lower():
                    subloc = "Cervical / Neck"
                elif "inguinal" in name.lower():
                    subloc = "Inguinal"
                elif "axillary" in name.lower() or "rotter" in name.lower():
                    subloc = "Axillary"
                elif "iliac" in name.lower():
                    subloc = "External Iliac" if "external" in name.lower() else "Internal Iliac" if "internal" in name.lower() else "Common Iliac"
                elif "obturator" in name.lower():
                    subloc = "Obturator"
                elif "paraaortic" in name.lower() or "aortic" in name.lower():
                    subloc = "Retroperitoneal / Para-Aortic"

                ontology[name] = {
                    "detailed": name.replace("_", " ") + " Lymph Node",
                    "generalized": "",
                    "side": side,
                    "anatomic_location": "Distant Lymph Node (Common Iliac, Retroperitoneal, Inguinal, Supraclavicular, Axillary)"
                                         if "common_iliac" not in name.lower() else "Pelvic Lymph Node",
                    "lesion_type": "Lymph Node Meta",
                    "anatomical_sublocation": subloc
                }
            elif any(kw in name.lower() for kw in [
                "vertebrae", "rib_", "femur", "hip_", "sacrum", "scapula",
                "humerus", "clavicula", "skull", "sternum"
            ]):
                ontology[name] = {
                    "detailed": name.replace("_", " "),
                    "generalized": "",
                    "side": "Left" if "_left" in name else "Right" if "_right" in name else "",
                    "anatomic_location": "Bone",
                    "lesion_type": "Bone Meta",
                    "anatomical_sublocation": "Axial Skeleton" if "vertebrae" in name or "sacrum" in name or "rib" in name else "Appendicular Skeleton"
                }
            else:
                # Generic organ/muscle
                ontology[name] = {
                    "detailed": name.replace("_", " "),
                    "generalized": "",
                    "side": "Left" if "_left" in name else "Right" if "_right" in name else "",
                    "anatomic_location": "Solid Organ / Viscera",
                    "lesion_type": "Organ Meta",
                    "anatomical_sublocation": "N/A (General Organ)"
                }
        print(f"  Added {len(missing)} entries to ontology")
    else:
        print("  All labels have ontology entries - no updates needed")

    return ontology


def create_student_h5(data, output_path):
    """Create the student PET/CT-only HDF5 file."""
    print(f"[4/6] Creating student HDF5: {output_path}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Remove existing file if present
    if os.path.exists(output_path):
        os.remove(output_path)
        print(f"  Removed existing file")

    shape = data["ct"].shape
    print(f"  Volume shape: {shape}")

    with h5py.File(output_path, "w") as h5:
        # ================================================================
        # BASELINE Group (TP0 — blank, for student practice)
        # ================================================================
        g_base = h5.create_group("BASELINE")
        ds_ct0 = g_base.create_dataset("Fixed_CT_Volume_0.nii.gz", data=data["ct"],
                              compression="gzip", compression_opts=4)
        # Copy attributes (spacing, origin, direction) from source CT
        for attr_name, attr_val in data["ct_attrs"].items():
            ds_ct0.attrs[attr_name] = attr_val
        ds_pet0 = g_base.create_dataset("SUV_PET_Image_0.nii.gz", data=data["pet"],
                              compression="gzip", compression_opts=4)
        for attr_name, attr_val in data["pet_attrs"].items():
            ds_pet0.attrs[attr_name] = attr_val
        # BLANK mask — all zeros (student will annotate this)
        blank_mask = np.zeros(shape, dtype=np.int16)
        ds_mask0 = g_base.create_dataset("PET_Lesions_0.nii.gz", data=blank_mask,
                              compression="gzip", compression_opts=4)
        for attr_name, attr_val in data["mask_attrs"].items():
            ds_mask0.attrs[attr_name] = attr_val
        ds_mask0e = g_base.create_dataset("PET_Lesions_0.nii.gz_expert", data=blank_mask,
                              compression="gzip", compression_opts=4)
        for attr_name, attr_val in data["mask_attrs"].items():
            ds_mask0e.attrs[attr_name] = attr_val
        g_base.create_dataset("diffusivity", data=data["diffusivity"],
                              compression="gzip", compression_opts=4)
        g_base.create_dataset("max_anatomy.nii.gz", data=data["max_anatomy"],
                              compression="gzip", compression_opts=4)
        g_base.create_dataset("max_anatomy.nii.gz_expert", data=data["max_anatomy_expert"],
                              compression="gzip", compression_opts=4)
        g_base.create_dataset("skellytour.nii.gz", data=data["skellytour"],
                              compression="gzip", compression_opts=4)
        print("  Created BASELINE (blank mask)")

        # ================================================================
        # TP1 Group (Follow-up — expert reference)
        # ================================================================
        g_tp1 = h5.create_group("TFM_Transform_FollowUp_to_Baseline_1.tfm")
        ds_ct1 = g_tp1.create_dataset("Fixed_CT_Volume_1.nii.gz", data=data["ct"],
                             compression="gzip", compression_opts=4)
        for attr_name, attr_val in data["ct_attrs"].items():
            ds_ct1.attrs[attr_name] = attr_val
        ds_pet1 = g_tp1.create_dataset("SUV_PET_Image_1.nii.gz", data=data["pet"],
                             compression="gzip", compression_opts=4)
        for attr_name, attr_val in data["pet_attrs"].items():
            ds_pet1.attrs[attr_name] = attr_val
        # Expert mask WITH annotations
        ds_mask1 = g_tp1.create_dataset("PET_Lesions_1.nii.gz", data=data["mask_auto"],
                             compression="gzip", compression_opts=4)
        for attr_name, attr_val in data["mask_attrs"].items():
            ds_mask1.attrs[attr_name] = attr_val
        ds_mask1e = g_tp1.create_dataset("PET_Lesions_1.nii.gz_expert", data=data["mask_expert"],
                             compression="gzip", compression_opts=4)
        for attr_name, attr_val in data["mask_attrs"].items():
            ds_mask1e.attrs[attr_name] = attr_val
        g_tp1.create_dataset("diffusivity", data=data["diffusivity"],
                             compression="gzip", compression_opts=4)
        g_tp1.create_dataset("max_anatomy.nii.gz", data=data["max_anatomy"],
                             compression="gzip", compression_opts=4)
        print("  Created TP1 (expert reference mask)")

        # ================================================================
        # ATLAS Group
        # ================================================================
        g_atlas = h5.create_group("ATLAS")
        g_atlas.create_dataset("bone_atlas", data=data["atlas_bone"],
                               compression="gzip", compression_opts=4)
        g_atlas.create_dataset("max_anatomy", data=data["atlas_max_anatomy"],
                               compression="gzip", compression_opts=4)
        g_atlas.create_dataset("max_anatomy_expert", data=data["atlas_max_anatomy_expert"],
                               compression="gzip", compression_opts=4)
        g_atlas.create_dataset("skellytour", data=data["atlas_skellytour"],
                               compression="gzip", compression_opts=4)
        print("  Created ATLAS")

        # ================================================================
        # CENTROIDS Group
        # ================================================================
        g_cent = h5.create_group("CENTROIDS")
        # TP0 centroids are empty (blank mask)
        # TP1 centroids — copy from source TP0 but rename tp0 → tp1
        for k, v in data["centroids_tp0"].items():
            # Source centroids: tp0_lid1 → tp1_lid1 for reference TP
            new_key = k.replace("tp0_", "tp1_")
            g_cent.create_dataset(new_key, data=v)
        print(f"  Created CENTROIDS ({len(data['centroids_tp0'])} TP1 entries)")

        # ================================================================
        # BONE_SUBSEG Group
        # ================================================================
        g_bone = h5.create_group("BONE_SUBSEG")
        # Copy TP0 bone subsegs as TP1
        for k, v in data["bone_subsegs_tp0"].items():
            new_key = k.replace("PET_Lesions_0_", "PET_Lesions_1_").replace("tp_0_", "tp_1_")
            g_bone.create_dataset(new_key, data=v)
        print(f"  Created BONE_SUBSEG ({len(data['bone_subsegs_tp0'])} entries)")

        # ================================================================
        # _meta_ Group
        # ================================================================
        g_meta = h5.create_group("_meta_")

        source_meta = data["meta"]
        ma_labels = json.loads(source_meta["max_anatomy_labels.json"])

        # --- metadata.json ---
        new_metadata = build_metadata_for_student(source_meta)
        g_meta.create_dataset("metadata.json",
                              data=json.dumps(new_metadata, indent=2, ensure_ascii=False))

        # --- scene_hierarchy.json ---
        hierarchy = build_scene_hierarchy()
        hierarchy = populate_scene_segments(hierarchy, data["mask_expert"])
        g_meta.create_dataset("scene_hierarchy.json",
                              data=json.dumps(hierarchy, indent=2, ensure_ascii=False))

        # --- segment_names.json ---
        seg_names = build_segment_names_for_tp1(source_meta)
        g_meta.create_dataset("segment_names.json",
                              data=json.dumps(seg_names, indent=2, ensure_ascii=False))

        # --- organ_mapping ---
        org_map = build_organ_mapping_for_tp1(source_meta)
        g_meta.create_dataset("organ_mapping",
                              data=json.dumps(org_map, indent=2, ensure_ascii=False))

        # --- lesion_annotations.json ---
        # TP0 = empty, TP1 = source annotations
        la = build_lesion_annotations_for_tp1(source_meta)
        g_meta.create_dataset("lesion_annotations.json",
                              data=json.dumps(la, indent=2, ensure_ascii=False))

        # --- matches.json ---
        matches = build_matches_for_tp1(source_meta)
        g_meta.create_dataset("matches.json",
                              data=json.dumps(matches, indent=2, ensure_ascii=False))

        # --- max_anatomy_labels.json ---
        g_meta.create_dataset("max_anatomy_labels.json",
                              data=source_meta["max_anatomy_labels.json"])

        # --- prostate_anatomy_labels.json ---
        if "prostate_anatomy_labels.json" in source_meta:
            g_meta.create_dataset("prostate_anatomy_labels.json",
                                  data=source_meta["prostate_anatomy_labels.json"])

        # --- bone_label_ids ---
        if "bone_label_ids" in source_meta:
            bone_ids = source_meta["bone_label_ids"]
            g_meta.create_dataset("bone_label_ids", data=bone_ids)

        # --- display_config.json ---
        display_config = {
            "label_opacity": 0.75,
            "heatgdt_ramp_speed": 4.0,
            "pet_ct_blend": 0.24
        }
        g_meta.create_dataset("display_config.json",
                              data=json.dumps(display_config, indent=2))

        # --- custom_options.json ---
        if "custom_options.json" in source_meta:
            g_meta.create_dataset("custom_options.json",
                                  data=source_meta["custom_options.json"])

        # --- metadata_presets.json ---
        if "metadata_presets.json" in source_meta:
            g_meta.create_dataset("metadata_presets.json",
                                  data=source_meta["metadata_presets.json"])

        # --- lymph_node_validations.json ---
        if "lymph_node_validations.json" in source_meta:
            g_meta.create_dataset("lymph_node_validations.json",
                                  data=source_meta["lymph_node_validations.json"])

        # --- preflipped ---
        g_meta.create_dataset("preflipped", data=np.int64(1))

        print("  Created _meta_ group with all metadata")

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Output file size: {file_size_mb:.1f} MB")


def verify_student_h5(output_path):
    """Verify the student dataset is correct."""
    print(f"[5/6] Verifying student dataset: {output_path}")

    f = h5py.File(output_path, "r")

    # Check top-level groups
    expected_groups = {"ATLAS", "BASELINE", "BONE_SUBSEG", "CENTROIDS",
                       "TFM_Transform_FollowUp_to_Baseline_1.tfm", "_meta_"}
    actual_groups = set(f.keys())
    assert expected_groups == actual_groups, \
        f"Group mismatch: expected {expected_groups}, got {actual_groups}"
    print(f"  Top-level groups OK: {sorted(actual_groups)}")

    # Check BASELINE (TP0 — blank)
    base = f["BASELINE"]
    assert "Fixed_CT_Volume_0.nii.gz" in base
    assert "SUV_PET_Image_0.nii.gz" in base
    assert "PET_Lesions_0.nii.gz_expert" in base
    mask0 = base["PET_Lesions_0.nii.gz_expert"][:]
    unique_labels = np.unique(mask0)
    assert len(unique_labels) == 1 and unique_labels[0] == 0, \
        f"TP0 mask should be all zeros, got unique labels: {unique_labels}"
    print(f"  BASELINE OK: CT+PET volumes present, mask is blank (all zeros)")

    # Check TP1 (reference with annotations)
    tp1 = f["TFM_Transform_FollowUp_to_Baseline_1.tfm"]
    assert "Fixed_CT_Volume_1.nii.gz" in tp1
    assert "SUV_PET_Image_1.nii.gz" in tp1
    assert "PET_Lesions_1.nii.gz_expert" in tp1
    mask1 = tp1["PET_Lesions_1.nii.gz_expert"][:]
    unique_labels_1 = np.unique(mask1)
    assert len(unique_labels_1) > 1, \
        f"TP1 mask should have annotations, got only: {unique_labels_1}"
    n_lesions = len([l for l in unique_labels_1 if l != 0])
    print(f"  TP1 OK: CT+PET volumes present, mask has {n_lesions} lesion labels")

    # Check CT and PET are identical between TPs (same scan)
    ct0 = base["Fixed_CT_Volume_0.nii.gz"][:]
    ct1 = tp1["Fixed_CT_Volume_1.nii.gz"][:]
    assert np.array_equal(ct0, ct1), "CT volumes should be identical"
    pet0 = base["SUV_PET_Image_0.nii.gz"][:]
    pet1 = tp1["SUV_PET_Image_1.nii.gz"][:]
    assert np.array_equal(pet0, pet1), "PET volumes should be identical"
    print("  CT and PET volumes identical between TPs - OK")

    # Check metadata
    meta = f["_meta_"]
    metadata = json.loads(meta["metadata.json"][()])
    assert len(metadata) == 3, f"Should have 3 metadata entries (patient + 2 studies)"
    assert metadata[0]["Modalities"] == ["CT", "PET"]
    print(f"  metadata.json OK: {metadata[0]['PatientID']}, modalities={metadata[0]['Modalities']}")

    # Check scene hierarchy
    sh = json.loads(meta["scene_hierarchy.json"][()])
    print(f"  scene_hierarchy.json OK: {len(sh)} nodes")

    # Check segment names
    sn = json.loads(meta["segment_names.json"][()])
    assert "0" in sn and "1" in sn
    assert sn["0"] == {} or len(sn["0"]) == 0, "TP0 should have no segment names"
    assert len(sn["1"]) > 0, "TP1 should have segment names"
    print(f"  segment_names OK: TP0={len(sn['0'])} names, TP1={len(sn['1'])} names")

    # Check ATLAS
    atlas = f["ATLAS"]
    assert "max_anatomy" in atlas
    assert "bone_atlas" in atlas
    print(f"  ATLAS OK: {list(atlas.keys())}")

    f.close()
    print("  ALL VERIFICATION CHECKS PASSED")


def refresh_ontology_file(ontology_path, max_anatomy_labels_json, also_copy_to=None):
    """Refresh the ontology JSON file and optionally copy to data dir."""
    ma_labels = json.loads(max_anatomy_labels_json)
    ontology = refresh_ontology_labels(ontology_path, ma_labels)

    # Write updated ontology
    with open(ontology_path, "w") as f:
        json.dump(ontology, f, indent=2, ensure_ascii=False)
    print(f"  Saved updated ontology to: {ontology_path}")

    if also_copy_to and os.path.exists(os.path.dirname(also_copy_to)):
        with open(also_copy_to, "w") as f:
            json.dump(ontology, f, indent=2, ensure_ascii=False)
        print(f"  Saved copy to: {also_copy_to}")

    return ontology


def main():
    print("=" * 70)
    print("  MedEye3d Student PET/CT Dataset Creator")
    print("=" * 70)
    print()

    # Step 1: Load source data
    data = load_source_data(SOURCE_H5)
    print()

    # Step 2: Build metadata
    print("[2/6] Building metadata for 2-TP PET/CT student dataset")
    print()

    # Step 3: Refresh ontology labels
    refresh_ontology_file(
        ONTOLOGY_PATH,
        data["meta"]["max_anatomy_labels.json"],
        also_copy_to=DATA_ONTOLOGY_PATH
    )
    print()

    # Step 4: Create the student HDF5
    create_student_h5(data, OUTPUT_H5)
    print()

    # Step 5: Verify
    verify_student_h5(OUTPUT_H5)
    print()

    # Step 6: Summary
    print("[6/6] DONE!")
    print(f"  Student dataset: {OUTPUT_H5}")
    print(f"  TP0 (BASELINE): PET/CT with BLANK mask - for student annotation practice")
    print(f"  TP1 (Reference): PET/CT with EXPERT mask - for student self-check")
    print(f"  Ontology refreshed: {ONTOLOGY_PATH}")
    print()
    print("  To open in MedEye3d, update run_interactive_mrb.sh to point to:")
    print(f"    {OUTPUT_H5}")


if __name__ == "__main__":
    main()
