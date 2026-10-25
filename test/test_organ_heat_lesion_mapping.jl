#!/usr/bin/env julia
"""
test_organ_heat_lesion_mapping.jl

Comprehensive test suite for the 5-organ relative anatomical mapping
and heat-kernel lesion growth pipeline:
1. Fast organ centroid extraction and HDF5 round-trip.
2. 5-closest-organ selection and Inverse Distance Weighting (IDW) translation.
3. Heat-kernel lesion diffusion growth with target volume matching.
4. Boolean AND collision detection logic (linking vs. synthetic generation).
5. Patient 6 integration verification (matches.json, HDF5 datasets, SPECT mirroring).
"""

using Test
using HDF5
using JSON
using LinearAlgebra

using MedEye3d
using MedEye3d.OrganHeatLesionTracker

@testset "5-Organ Anatomical Mapping & Heat-Kernel Pipeline" begin

    # =========================================================================
    # Group 1: Organ Centroid Extraction & IDW Translation Math
    # =========================================================================
    @testset "1. Organ Centroid Extraction & IDW Translation Math" begin
        # Create synthetic max_anatomy volume (50x50x50) with 6 known organs
        ma_vol = zeros(UInt16, 50, 50, 50)
        
        # Organ 1: centered at [10, 10, 10]
        ma_vol[8:12, 8:12, 8:12] .= 1
        # Organ 2: centered at [40, 10, 10]
        ma_vol[38:42, 8:12, 8:12] .= 2
        # Organ 3: centered at [10, 40, 10]
        ma_vol[8:12, 38:42, 8:12] .= 3
        # Organ 4: centered at [40, 40, 10]
        ma_vol[38:42, 38:42, 8:12] .= 4
        # Organ 5: centered at [25, 25, 40]
        ma_vol[23:27, 23:27, 38:42] .= 5
        # Organ 6: far away at [48, 48, 48]
        ma_vol[47:49, 47:49, 47:49] .= 6

        centroids_A = compute_organ_centroids(ma_vol)
        @test length(centroids_A) == 6
        @test isapprox(centroids_A[1], [10.0, 10.0, 10.0], atol=1e-3)
        @test isapprox(centroids_A[5], [25.0, 25.0, 40.0], atol=1e-3)

        # Lesion at [12, 12, 12] (close to organ 1)
        lesion_A = [12.0, 12.0, 12.0]
        closest_5 = find_5_closest_organs(lesion_A, centroids_A, centroids_A)
        @test length(closest_5) == 5
        @test closest_5[1] == 1 # Organ 1 must be the closest!
        @test !(6 in closest_5) # Organ 6 is farthest, must be excluded

        # Simulate translation at TP B: all organs move by +5 in X, -2 in Y, +3 in Z
        shift_vector = [5.0, -2.0, 3.0]
        centroids_B = Dict{Int, Vector{Float64}}()
        for (lid, c) in centroids_A
            centroids_B[lid] = c .+ shift_vector
        end

        # Test IDW translation
        lesion_B = translate_lesion_center(lesion_A, closest_5, centroids_A, centroids_B)
        expected_B = lesion_A .+ shift_vector
        @test isapprox(lesion_B, expected_B, atol=1e-3)
    end

    # =========================================================================
    # Group 2: Heat-Kernel Diffusion Growth to Target Volume
    # =========================================================================
    @testset "2. Heat-Kernel Lesion Growth & Volume Matching" begin
        # Uniform diffusivity volume 40x40x40
        D = ones(Float32, 40, 40, 40)
        center = [20.0, 20.0, 20.0]
        target_vol = 150 # target voxels

        patch_mask, bbox = grow_heat_kernel_lesion(D, center, target_vol; patch_size=32, K=15)
        
        segmented_voxels = count(patch_mask .> 0)
        # Verify within 15% of target volume
        @test abs(segmented_voxels - target_vol) / target_vol < 0.15
        @test segmented_voxels > 0

        # Verify mask is contiguous around center
        cx_p = round(Int, center[1]) - bbox[1] + 1
        cy_p = round(Int, center[2]) - bbox[3] + 1
        cz_p = round(Int, center[3]) - bbox[5] + 1
        @test patch_mask[cx_p, cy_p, cz_p] == 1
    end

    # =========================================================================
    # Group 3: Boolean AND Collision Detection Logic
    # =========================================================================
    @testset "3. Boolean AND Collision Detection" begin
        target_mask = zeros(Int16, 50, 50, 50)
        # Existing original lesion #3 at [20:25, 20:25, 20:25]
        target_mask[20:25, 20:25, 20:25] .= 3

        # Case A: Heat patch that overlaps lesion #3
        heat_patch_overlap = zeros(UInt8, 10, 10, 10)
        heat_patch_overlap[2:6, 2:6, 2:6] .= 1
        bbox_overlap = (19, 28, 19, 28, 19, 28)

        best_cand, overlap_count = detect_overlap_and_link(heat_patch_overlap, bbox_overlap, target_mask; max_original_id=10)
        @test best_cand == 3
        @test overlap_count > 0

        # Case B: Heat patch that does NOT overlap (empty region)
        bbox_no_overlap = (35, 44, 35, 44, 35, 44)
        best_cand_empty, overlap_empty = detect_overlap_and_link(heat_patch_overlap, bbox_no_overlap, target_mask; max_original_id=10)
        @test best_cand_empty == 0
        @test overlap_empty == 0
    end

    # =========================================================================
    # Group 4: Patient 6 Integration Verification
    # =========================================================================
    @testset "4. Patient 6 Real Case Integration" begin
        h5_path = "/workspaces/MedEye3d.jl/data/pat_6_files/preprocessed_volumes.h5"
        matches_path = "/workspaces/MedEye3d.jl/data/pat_6_files/matches.json"
        report_path = "/workspaces/MedEye3d.jl/data/pat_6_files/lesion_tracking_report.json"

        @test isfile(h5_path)
        @test isfile(matches_path)
        @test isfile(report_path)

        # 1. Verify organ centroids in HDF5
        h5open(h5_path, "r") do h5
            @test haskey(h5, "_meta_")
            @test haskey(h5["_meta_"], "organ_centroids.json")
            @test haskey(h5, "ORGAN_CENTROIDS")
            @test haskey(h5, "CENTROIDS")
            
            # Verify organ_centroids.json has data for all 4 TPs
            json_str = read(h5["_meta_/organ_centroids.json"])
            data = JSON.parse(json_str)
            @test haskey(data, "tp_0")
            @test haskey(data, "tp_1")
            @test haskey(data, "tp_2")
            @test haskey(data, "tp_3")
            @test length(data["tp_0"]) >= 250
        end

        # 2. Verify matches.json
        matches_data = JSON.parse(read(matches_path, String))
        @test length(matches_data) > 0
        
        # Check that we have both ORGAN_HEAT_MATCH and HEAT_KERNEL_SYNTHETIC
        all_child_types = String[]
        all_synth_volumes = Float64[]
        for g in matches_data
            for c in get(g, "children", [])
                mtype = get(c, "match_type", "")
                push!(all_child_types, mtype)
                if mtype == "HEAT_KERNEL_SYNTHETIC"
                    push!(all_synth_volumes, Float64(c["volume_mm3"]))
                end
            end
        end

        @test "ORGAN_HEAT_MATCH" in all_child_types
        @test "HEAT_KERNEL_SYNTHETIC" in all_child_types
        @test "SPECT_ORGAN_HEAT_MAPPED" in all_child_types
        @test !("SPECT_MIRROR" in all_child_types)

        # Verify that SPECT lesions are not an identical copy of PET lesions
        h5open(h5_path, "r") do h5
            pet0 = read(h5["BASELINE/PET_Lesions_0.nii.gz"])
            spect0 = read(h5["TFM_Transform_SPECT_to_Baseline_0.tfm/SPECT_Lesions_0.nii.gz"])
            @test count(spect0 .> 0) > 0
            @test !isequal(pet0, spect0)
        end

        # Verify that synthetic volumes are dynamic (NOT constant 1343.63)
        @test length(all_synth_volumes) > 0
        @test length(unique(all_synth_volumes)) > 5
        @test minimum(all_synth_volumes) < 200.0
        @test maximum(all_synth_volumes) > 1000.0

        # 3. Verify lesion_tracking_report.json
        report_data = JSON.parse(read(report_path, String))
        @test length(report_data) >= 400
        
        # Check that synthetic lesions have physiological residual background SUV (< 3.0)
        synth_suvs = [Float64(r["mean_intensity"]) for r in report_data if r["match_type"] == "HEAT_KERNEL_SYNTHETIC"]
        @test length(synth_suvs) > 0
        avg_synth_suv = sum(synth_suvs) / length(synth_suvs)
        @test avg_synth_suv < 3.0

        # 4. Verify physiological uptake exclusion (Parotid glands and Urinary bladder)
        h5open(h5_path, "r") do h5
            mapping = get_group_label_mapping(h5)
            # Verify across all PET timepoints
            for (tp, grp) in [(0, "BASELINE"), 
                              (1, "TFM_Transform_FollowUp_to_Baseline_1.tfm"),
                              (2, "TFM_Transform_FollowUp_to_Baseline_2.tfm"),
                              (3, "TFM_Transform_FollowUp_to_Baseline_3.tfm")]
                ma = read(h5["$grp/max_anatomy.nii.gz"])
                lbls = mapping[grp]
                excl = get_exclusion_organ_ids(lbls)
                @test !isempty(excl)
                mask = read(h5["$grp/PET_Lesions_$tp.nii.gz"])
                for eid in excl
                    overlap_vox = count((ma .== eid) .& (mask .> 0))
                    @test overlap_vox == 0
                end
            end

            # Verify across paired SPECT timepoints
            for grp in keys(h5)
                if occursin("TFM_Transform_SPECT", grp) && haskey(h5[grp], "max_anatomy.nii.gz")
                    ma = read(h5["$grp/max_anatomy.nii.gz"])
                    lbls = mapping[grp]
                    excl = get_exclusion_organ_ids(lbls)
                    @test !isempty(excl)
                    for k in keys(h5[grp])
                        if occursin("SPECT_Lesions", k) && !occursin("_expert", k)
                            mask = read(h5["$grp/$k"])
                            for eid in excl
                                overlap_vox = count((ma .== eid) .& (mask .> 0))
                                @test overlap_vox == 0
                            end
                        end
                    end
                end
            end
        end
    end

end
