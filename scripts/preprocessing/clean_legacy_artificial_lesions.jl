#!/usr/bin/env julia
"""
clean_legacy_artificial_lesions.jl

Cleanly removes legacy spherical artificial lesions and resets match groups
in Patient 6 files and preprocessed_volumes.h5:
1. Zeroes out legacy artificial labels in PET and SPECT lesion masks:
   - TP1: labels 46..54 (9 lesions)
   - TP2: labels 37..72 (29 lesions)
   - TP3: labels 56..107 (38 lesions)
2. Resets matches.json and _meta_/matches.json to an empty baseline.
3. Clears legacy CENTROIDS entries for artificial lesions.
"""

using HDF5
using JSON

function clean_legacy_artificial_lesions(h5_path::String)
    println("=" ^ 70)
    println("  CLEANING LEGACY ARTIFICIAL LESIONS: $h5_path")
    println("=" ^ 70)
    
    if !isfile(h5_path)
        error("HDF5 file not found: $h5_path")
    end

    # Define artificial label ranges per TP
    # TP0: none
    # TP1: 46..9999 (original is 1..45)
    # TP2: 37..9999 (original is 1..36)
    # TP3: 55..9999 (original is max 54)
    art_ranges = Dict(
        "TFM_Transform_FollowUp_to_Baseline_1.tfm" => (46, 9999),
        "TFM_Transform_FollowUp_to_Baseline_2.tfm" => (37, 9999),
        "TFM_Transform_FollowUp_to_Baseline_3.tfm" => (55, 9999),
        "TFM_Transform_SPECT_to_Baseline_1.tfm" => (46, 9999),
        "TFM_Transform_SPECT_to_Baseline_2.tfm" => (37, 9999),
        "TFM_Transform_SPECT_to_Baseline_4.tfm" => (55, 9999)
    )

    h5open(h5_path, "r+") do h5
        # 1. Zero out artificial voxels in datasets
        for (grp_name, (min_art, max_art)) in art_ranges
            if haskey(h5, grp_name)
                grp = h5[grp_name]
                for ds_name in keys(grp)
                    if occursin("Lesions", ds_name)
                        arr = read(grp[ds_name])
                        art_voxels = count(v -> (v >= min_art && v <= max_art), arr)
                        if art_voxels > 0
                            arr[arr .>= min_art .&& arr .<= max_art] .= 0
                            # Overwrite dataset
                            delete_object(grp, ds_name)
                            grp[ds_name] = arr
                            println("  [Cleaned] $grp_name/$ds_name: zeroed $art_voxels voxels (range $min_art..$max_art)")
                        else
                            println("  [Clean] $grp_name/$ds_name: no artificial voxels found")
                        end
                    end
                end
            end
        end

        # 2. Reset matches in _meta_
        if haskey(h5, "_meta_")
            meta = h5["_meta_"]
            if haskey(meta, "matches.json")
                delete_object(meta, "matches.json")
            end
            meta["matches.json"] = "[]"
            println("  [Reset] _meta_/matches.json -> []")
        end

        # 3. Clean legacy artificial centroids in CENTROIDS
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

    # Also clean matches.json on disk if adjacent
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

    println("\n[Done] Legacy artificial lesions cleaned and match groups reset.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    target_h5 = length(ARGS) > 0 ? ARGS[1] : "/workspaces/MedEye3d.jl/data/pat_6_files/preprocessed_volumes.h5"
    clean_legacy_artificial_lesions(target_h5)
end
