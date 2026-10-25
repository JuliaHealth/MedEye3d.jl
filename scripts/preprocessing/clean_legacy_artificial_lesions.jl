#!/usr/bin/env julia
"""
clean_legacy_artificial_lesions.jl

Cleanly removes legacy spherical artificial lesions and resets match groups
in Patient 6 files and preprocessed_volumes.h5:
1. Zeroes out legacy artificial labels in PET and resets SPECT lesion masks:
   - TP1: labels >= 46
   - TP2: labels >= 37
   - TP3: labels >= 55
   - SPECT 0, 1, 2, 4: reset to zero (to be re-generated via 5-organ heat diffusion)
2. Resets matches.json and _meta_/matches.json to an empty baseline.
3. Clears legacy CENTROIDS entries for artificial lesions.
4. Cleans adjacent disk NIfTI files (PET_Lesions_*.nii.gz, SPECT_Lesions_*.nii.gz).
"""

using HDF5
using JSON
using Statistics
using MedEye3d
using MedEye3d.OrganHeatLesionTracker: get_group_label_mapping, get_exclusion_organ_ids, filter_exclusion_lesions!

function clean_legacy_artificial_lesions(h5_path::String)
    println("=" ^ 70)
    println("  CLEANING LEGACY ARTIFICIAL LESIONS: $h5_path")
    println("=" ^ 70)
    
    if !isfile(h5_path)
        error("HDF5 file not found: $h5_path")
    end

    # Define artificial label cutoffs per PET group
    # TP0: none (original is 1..31)
    # TP1: >= 46 (original is 1..45)
    # TP2: >= 37 (original is 1..36)
    # TP3: >= 55 (original is max 54)
    pet_cutoffs = Dict(
        "TFM_Transform_FollowUp_to_Baseline_1.tfm" => 46,
        "TFM_Transform_FollowUp_to_Baseline_2.tfm" => 37,
        "TFM_Transform_FollowUp_to_Baseline_3.tfm" => 55
    )

    spect_groups = [
        "TFM_Transform_SPECT_to_Baseline_0.tfm",
        "TFM_Transform_SPECT_to_Baseline_1.tfm",
        "TFM_Transform_SPECT_to_Baseline_2.tfm",
        "TFM_Transform_SPECT_to_Baseline_4.tfm"
    ]

    h5open(h5_path, "r+") do h5
        # 1. Clean PET datasets
        if haskey(h5, "BASELINE") && haskey(h5["BASELINE"], "PET_Lesions_0.nii.gz_expert")
            delete_object(h5["BASELINE"], "PET_Lesions_0.nii.gz_expert")
            println("  [Removed Stale Expert] BASELINE/PET_Lesions_0.nii.gz_expert")
        end

        for (grp_name, min_art) in pet_cutoffs
            if haskey(h5, grp_name)
                grp = h5[grp_name]
                for ds_name in collect(keys(grp))
                    if occursin("Lesions", ds_name)
                        if occursin("_expert", ds_name)
                            delete_object(grp, ds_name)
                            println("  [Removed Stale Expert] $grp_name/$ds_name")
                        else
                            arr = read(grp[ds_name])
                            art_voxels = count(v -> (v >= min_art), arr)
                            if art_voxels > 0
                                arr[arr .>= min_art] .= 0
                                delete_object(grp, ds_name)
                                grp[ds_name] = arr
                                println("  [Cleaned] $grp_name/$ds_name: zeroed $art_voxels voxels (values >= $min_art)")
                            else
                                println("  [Clean] $grp_name/$ds_name: no voxels >= $min_art found")
                            end
                        end
                    end
                end
            end
        end

        # 1.5 Filter out physiological uptake in parotid glands, lacrimal glands, and urinary bladder
        group_labels = get_group_label_mapping(h5)
        for grp_name in ["BASELINE", keys(pet_cutoffs)...]
            if haskey(h5, grp_name) && haskey(h5[grp_name], "max_anatomy.nii.gz")
                anat = read(h5["$grp_name/max_anatomy.nii.gz"])
                lbl_dict = get(group_labels, grp_name, Dict{String, String}())
                excl_ids = get_exclusion_organ_ids(lbl_dict)
                if !isempty(excl_ids)
                    for ds_name in collect(keys(h5[grp_name]))
                        if occursin("Lesions", ds_name)
                            mask = read(h5["$grp_name/$ds_name"])
                            rem_lids = filter_exclusion_lesions!(mask, anat, excl_ids)
                            if !isempty(rem_lids)
                                delete_object(h5[grp_name], ds_name)
                                h5["$grp_name/$ds_name"] = mask
                                println("  [Cleaned Physiological] $grp_name/$ds_name: removed lesions $rem_lids in parotid/lacrimal/bladder")
                            end
                        end
                    end
                end
            end
        end

        # 2. Reset SPECT datasets to zeros and remove stale expert masks
        for grp_name in spect_groups
            if haskey(h5, grp_name)
                grp = h5[grp_name]
                for ds_name in collect(keys(grp))
                    if occursin("Lesions", ds_name)
                        if occursin("_expert", ds_name)
                            delete_object(grp, ds_name)
                            println("  [Removed Stale Expert] $grp_name/$ds_name")
                        else
                            arr = read(grp[ds_name])
                            nonzeros = count(v -> v > 0, arr)
                            fill!(arr, 0)
                            delete_object(grp, ds_name)
                            grp[ds_name] = arr
                            println("  [Reset SPECT] $grp_name/$ds_name: reset $nonzeros voxels to 0")
                        end
                    end
                end
            end
        end

        # 3. Reset matches in _meta_
        if haskey(h5, "_meta_")
            meta = h5["_meta_"]
            if haskey(meta, "matches.json")
                delete_object(meta, "matches.json")
            end
            meta["matches.json"] = "[]"
            println("  [Reset] _meta_/matches.json -> []")
        end

        # 4. Clean legacy artificial centroids in CENTROIDS
        if haskey(h5, "CENTROIDS")
            c_grp = h5["CENTROIDS"]
            deleted_cents = 0
            for k in collect(keys(c_grp))
                m = match(r"^tp(\d+)_lid(\d+)$", k)
                if m !== nothing
                    tp = parse(Int, m.captures[1])
                    lid = parse(Int, m.captures[2])
                    is_art = false
                    if tp == 1 && lid >= 46; is_art = true; end
                    if tp == 2 && lid >= 37; is_art = true; end
                    if tp == 3 && lid >= 55; is_art = true; end
                    if is_art
                        delete_object(c_grp, k)
                        deleted_cents += 1
                    end
                end
            end
            println("  [Cleaned] CENTROIDS: removed $deleted_cents legacy artificial centroid entries")
        end
    end

    # 5. Clean adjacent files on disk
    data_dir = dirname(h5_path)
    matches_json_path = joinpath(data_dir, "matches.json")
    if isfile(matches_json_path)
        write(matches_json_path, "[]\n")
        println("  [Reset] $(basename(matches_json_path)) -> []")
    end

    report_path = joinpath(data_dir, "lesion_tracking_report.json")
    if isfile(report_path)
        write(report_path, "[]\n")
        println("  [Reset] $(basename(report_path)) -> []")
    end

    # Clean disk NIfTI files if python & nibabel available
    try
        py_clean_code = """
import os, sys, h5py
import nibabel as nib
import numpy as np

h5_path = sys.argv[1]
data_dir = sys.argv[2]

with h5py.File(h5_path, 'r') as h5:
    for tp, grp in [(0, 'BASELINE'), (1, 'TFM_Transform_FollowUp_to_Baseline_1.tfm'), 
                    (2, 'TFM_Transform_FollowUp_to_Baseline_2.tfm'), (3, 'TFM_Transform_FollowUp_to_Baseline_3.tfm')]:
        ds_name = f'PET_Lesions_{tp}.nii.gz'
        p = os.path.join(data_dir, ds_name)
        if grp in h5 and ds_name in h5[grp] and os.path.isfile(p):
            arr = h5[grp][ds_name][:]
            ref_img = nib.load(p)
            new_img = nib.Nifti1Image(arr.astype(np.int16), ref_img.affine, ref_img.header)
            nib.save(new_img, p)
            print(f'  [Synced Disk NIfTI] {ds_name}: synchronized with cleaned HDF5 mask')

for tp in [0, 1, 2, 4]:
    p = os.path.join(data_dir, f'SPECT_Lesions_{tp}.nii.gz')
    if os.path.isfile(p):
        img = nib.load(p)
        arr = img.get_fdata()
        cnt = np.count_nonzero(arr > 0)
        arr[:] = 0
        new_img = nib.Nifti1Image(arr.astype(np.int16), img.affine, img.header)
        nib.save(new_img, p)
        print(f'  [Reset Disk NIfTI] {os.path.basename(p)}: reset {cnt} voxels to 0')
"""
        cmd = `python3 -c $py_clean_code $h5_path $data_dir`
        run(cmd)
    catch e
        @warn "Failed cleaning disk NIfTI files with python nibabel: $e"
    end

    println("\n[Done] Legacy artificial lesions cleaned and match groups reset.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    target_h5 = length(ARGS) > 0 ? ARGS[1] : "/workspaces/MedEye3d.jl/data/pat_6_files/preprocessed_volumes.h5"
    clean_legacy_artificial_lesions(target_h5)
end
