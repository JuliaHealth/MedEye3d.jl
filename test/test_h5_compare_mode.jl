#!/usr/bin/env julia
# Test that all TPs in the fresh HDF5 can be loaded for compare mode
# This simulates what AppMain.load_single_tp_from_h5 does

using Pkg
Pkg.activate("/workspaces/MedEye3d.jl")
using HDF5
using Test
using Statistics

h5_path = "data/pat_6_files/preprocessed_volumes.h5"

@testset "HDF5 Compare Mode Integrity" begin

    @testset "File Structure" begin
        @test isfile(h5_path)
        h5open(h5_path, "r") do f
            @test haskey(f, "BASELINE")
            @test haskey(f, "_meta_")
            @test haskey(f, "_meta_/transforms")
            @test haskey(f, "_meta_/preflipped")
            
            # All 7 transform groups
            expected_groups = [
                "TFM_Transform_SPECT_to_Baseline_0.tfm",
                "TFM_Transform_FollowUp_to_Baseline_1.tfm",
                "TFM_Transform_SPECT_to_Baseline_1.tfm",
                "TFM_Transform_FollowUp_to_Baseline_2.tfm",
                "TFM_Transform_SPECT_to_Baseline_2.tfm",
                "TFM_Transform_FollowUp_to_Baseline_3.tfm",
                "TFM_Transform_SPECT_to_Baseline_4.tfm",
            ]
            for g in expected_groups
                @test haskey(f, g)
            end
        end
    end
    
    @testset "Load Each TP Volume" begin
        h5open(h5_path, "r") do f
            is_pf = haskey(f, "_meta_/preflipped") && read(f["_meta_/preflipped"]) == 1
            println("  preflipped = $is_pf")
            
            all_groups = [
                ("BASELINE", "Fixed_CT_Volume_0.nii.gz", "SUV_PET_Image_0.nii.gz", "PET_Lesions_0.nii.gz", "PET"),
                ("TFM_Transform_SPECT_to_Baseline_0.tfm", "SPECT_CT_Volume_0.nii.gz", "SPECT_NM_Vendor_Volume_0.nii.gz", "SPECT_Lesions_0.nii.gz", "SPECT"),
                ("TFM_Transform_FollowUp_to_Baseline_1.tfm", "Fixed_CT_Volume_1.nii.gz", "SUV_PET_Image_1.nii.gz", "PET_Lesions_1.nii.gz", "PET"),
                ("TFM_Transform_SPECT_to_Baseline_1.tfm", "SPECT_CT_Volume_1.nii.gz", "SPECT_NM_Vendor_Volume_1.nii.gz", "SPECT_Lesions_1.nii.gz", "SPECT"),
                ("TFM_Transform_FollowUp_to_Baseline_2.tfm", "Fixed_CT_Volume_2.nii.gz", "SUV_PET_Image_2.nii.gz", "PET_Lesions_2.nii.gz", "PET"),
                ("TFM_Transform_SPECT_to_Baseline_2.tfm", "SPECT_CT_Volume_2.nii.gz", "SPECT_NM_Vendor_Volume_2.nii.gz", "SPECT_Lesions_2.nii.gz", "SPECT"),
                ("TFM_Transform_FollowUp_to_Baseline_3.tfm", "Fixed_CT_Volume_3.nii.gz", "SUV_PET_Image_3.nii.gz", "PET_Lesions_3.nii.gz", "PET"),
                ("TFM_Transform_SPECT_to_Baseline_4.tfm", "SPECT_CT_Volume_4.nii.gz", "SPECT_NM_Vendor_Volume_4.nii.gz", "SPECT_Lesions_4.nii.gz", "SPECT"),
            ]
            
            for (i, (grp, ct_fn, pet_fn, mask_fn, modality)) in enumerate(all_groups)
                tp_i = i - 1
                @testset "TP $tp_i ($grp)" begin
                    @test haskey(f, grp)
                    
                    # Read CT
                    ct_path = "$grp/$ct_fn"
                    @test haskey(f, ct_path)
                    ct_vol = Float32.(read(f[ct_path]))
                    @test size(ct_vol) == (512, 512, 326)
                    @test !any(isnan, ct_vol)
                    println("    CT: min=$(minimum(ct_vol)), max=$(maximum(ct_vol)), mean=$(round(mean(ct_vol), digits=1))")
                    
                    # Read PET/NM
                    pet_path = "$grp/$pet_fn"
                    @test haskey(f, pet_path)
                    pet_vol = Float32.(read(f[pet_path]))
                    @test size(pet_vol) == (512, 512, 326)
                    @test !any(isnan, pet_vol)
                    
                    # SPECT scaling (same as AppMain)
                    if modality == "SPECT"
                        pos_dat = pet_vol[pet_vol .> 0]
                        p99 = isempty(pos_dat) ? 1.0f0 : Float32(quantile(pos_dat, 0.99))
                        scale_factor = 8.0f0 / max(p99, 1.0f0)
                        pet_vol_scaled = max.(0.0f0, pet_vol .* scale_factor)
                        println("    NM: p99=$p99, scale=$scale_factor, scaled_max=$(maximum(pet_vol_scaled))")
                    else
                        println("    PET: min=$(minimum(pet_vol)), max=$(maximum(pet_vol)), mean=$(round(mean(pet_vol), digits=3))")
                    end
                    
                    # Read mask
                    mask_path = "$grp/$mask_fn"
                    @test haskey(f, mask_path)
                    mask_vol = read(f[mask_path])
                    @test size(mask_vol) == (512, 512, 326)
                    n_nonzero = count(x -> x != 0, mask_vol)
                    println("    Mask: nonzero=$n_nonzero, unique_labels=$(sort(unique(mask_vol))[1:min(5,end)])")
                    
                    # Read anatomy
                    anat_path = "$grp/max_anatomy.nii.gz"
                    @test haskey(f, anat_path)
                    anat_vol = read(f[anat_path])
                    @test size(anat_vol) == (512, 512, 326)
                    println("    Anatomy: unique_count=$(length(unique(anat_vol)))")
                    
                    # Check expert mask for BASELINE
                    if grp == "BASELINE"
                        expert_path = "$grp/$(mask_fn)_expert"
                        @test haskey(f, expert_path)
                        expert_vol = read(f[expert_path])
                        @test size(expert_vol) == (512, 512, 326)
                        n_expert = count(x -> x != 0, expert_vol)
                        println("    Expert: nonzero=$n_expert")
                    end
                end
            end
        end
    end
    
    @testset "Metadata Integrity" begin
        h5open(h5_path, "r") do f
            # Transforms stored
            @test haskey(f, "_meta_/transforms")
            tg = f["_meta_/transforms"]
            @test length(keys(tg)) == 7  # 7 transforms (BASELINE has none)
            
            # Segment names
            @test haskey(f, "_meta_/segment_names.json")
            seg_names = String(read(f["_meta_/segment_names.json"]))
            @test length(seg_names) > 100
            
            # Scene hierarchy
            @test haskey(f, "_meta_/scene_hierarchy.json")
            hier = String(read(f["_meta_/scene_hierarchy.json"]))
            @test length(hier) > 50
            
            # Organ mapping (stored as JSON string dataset)
            @test haskey(f, "_meta_/organ_mapping")
            om_str = read(f["_meta_/organ_mapping"])
            @test isa(om_str, String)
            @test length(om_str) > 100
            println("  organ_mapping: $(length(om_str)) chars")
        end
    end
    
    @testset "Diffusivity Status" begin
        h5open(h5_path, "r") do f
            has_baseline = haskey(f, "BASELINE/diffusivity")
            @test has_baseline
            
            if has_baseline
                d = read(f["BASELINE/diffusivity"])
                @test size(d) == (512, 512, 326)
                @test eltype(d) == Float32
                println("  BASELINE diffusivity: min=$(minimum(d)), max=$(maximum(d)), mean=$(round(mean(d), digits=4))")
            end
            
            # Count how many TPs have diffusivity (in-progress computation)
            all_groups = filter(k -> k != "_meta_", collect(keys(f)))
            n_with = count(g -> haskey(f, "$g/diffusivity"), all_groups)
            println("  TPs with diffusivity: $n_with / $(length(all_groups))")
        end
    end
end
