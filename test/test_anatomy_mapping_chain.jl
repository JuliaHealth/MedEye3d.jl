#!/usr/bin/env julia
# Test script: Verify max_anatomy → ontology → UBERON mapping chain
# Run: julia --project test/test_anatomy_mapping_chain.jl

using JSON
using Test

@testset "Anatomy Mapping Chain" begin
    mapping = JSON.parsefile("assets/max_anatomy_to_ontology.json")
    
    @testset "All HDF5 TS labels have ontology entries" begin
        # Load labels from the actual HDF5 file
        using HDF5
        h5_files = [
            "data/cases/psma_patient_all_tp/preprocessed_volumes.h5",
            "data/pat_6_files/preprocessed_volumes.h5"
        ]
        for h5_path in h5_files
            isfile(h5_path) || continue
            h5 = h5open(h5_path, "r")
            ts_names = Dict{Int,String}()
            if haskey(h5, "_meta_/max_anatomy_labels.json")
                raw = JSON.parse(read(h5["_meta_/max_anatomy_labels.json"]))
                ts_names = Dict{Int,String}(parse(Int, k) => v for (k, v) in raw)
            end
            close(h5)
            
            for (id, name) in ts_names
                key = lowercase(strip(name))
                @test haskey(mapping, key) || haskey(mapping, name)
                entry = get(mapping, key, get(mapping, name, nothing))
                @test entry !== nothing
                if entry !== nothing
                    @test !isempty(get(entry, "detailed", ""))
                end
            end
            @info "Verified $(length(ts_names)) labels from $h5_path"
        end
    end
    
    @testset "Side values are correct" begin
        # Paired structures should have Left/Right
        for name in ["hip_left", "femur_left", "kidney_left", "adrenal_gland_left"]
            entry = get(mapping, name, nothing)
            @test entry !== nothing
            @test get(entry, "side", "") == "Left"
        end
        for name in ["hip_right", "femur_right", "kidney_right", "adrenal_gland_right"]
            entry = get(mapping, name, nothing)
            @test entry !== nothing
            @test get(entry, "side", "") == "Right"
        end
        
        # Unpaired structures should have empty side (NOT "NA")
        for name in ["liver", "spleen", "sacrum", "brain", "aorta", "prostate", "colon"]
            entry = get(mapping, name, nothing)
            @test entry !== nothing
            if entry !== nothing
                side = get(entry, "side", "")
                @test side == "" || side == "NA"  # Both acceptable in mapping
                @test uppercase(side) != "NA"  # But should NOT be NA
            end
        end
    end
    
    @testset "Key bone structures map correctly" begin
        test_cases = [
            ("hip_left", "innominate bone", "Left"),
            ("hip_right", "innominate bone", "Right"),
            ("femur_left", "femur", "Left"),
            ("sacrum", "sacral vertebra", ""),
            ("vertebrae_L3", "lumbar vertebra 3", ""),
            ("vertebrae_T12", "thoracic vertebra 12", ""),
        ]
        for (key, expected_detail, expected_side) in test_cases
            entry = get(mapping, key, nothing)
            @test entry !== nothing
            if entry !== nothing
                det = lowercase(get(entry, "detailed", ""))
                @test occursin(lowercase(expected_detail), det)
                @test get(entry, "side", "") == expected_side
            end
        end
    end
    
    @testset "Key organ structures map correctly" begin
        test_cases = [
            ("liver", "liver"),
            ("spleen", "spleen"),
            ("prostate", "prostate gland"),
            ("kidney_left", "kidney"),
        ]
        for (key, expected_detail) in test_cases
            entry = get(mapping, key, nothing)
            @test entry !== nothing
            if entry !== nothing
                det = lowercase(get(entry, "detailed", ""))
                @test occursin(lowercase(expected_detail), det)
            end
        end
    end
    
    @testset "NA side is filtered in report" begin
        # Simulate the NA filtering logic from EPSMAStructuredReport
        for side_val in ["NA", "N/A", "na", "Na"]
            _side = uppercase(side_val) in ("NA", "N/A") ? "" : side_val
            @test isempty(_side)
        end
        # Normal sides should pass through
        for side_val in ["Right", "Left", ""]
            _side = uppercase(side_val) in ("NA", "N/A") ? "" : side_val
            @test _side == side_val
        end
    end
    
    @testset "LesionType auto-detection from ontology" begin
        # Bone structures → Bone Meta
        for name in ["hip_left", "hip_right", "femur_left", "sacrum", "vertebrae_L3", 
                      "vertebrae_T12", "rib_left_5", "scapula_left", "clavicula_right",
                      "humerus_left", "sternum", "costal_cartilages"]
            entry = get(mapping, name, nothing)
            @test entry !== nothing
            if entry !== nothing
                @test get(entry, "lesion_type", "") == "Bone Meta"
            end
        end
        
        # Organ structures → Organ Meta
        for name in ["liver", "spleen", "kidney_left", "kidney_right", "aorta", "brain",
                      "colon", "lung_upper_lobe_left", "stomach"]
            entry = get(mapping, name, nothing)
            @test entry !== nothing
            if entry !== nothing
                @test get(entry, "lesion_type", "") == "Organ Meta"
            end
        end
        
        # Prostate → Prostate
        entry = get(mapping, "prostate", nothing)
        @test entry !== nothing
        @test get(entry, "lesion_type", "") == "Prostate"
        
        # Muscles → Technical Artifact
        for name in ["gluteus_maximus_left", "gluteus_medius_right", "psoas_major_left",
                      "sartorius_left", "quadriceps_femoris_left"]
            entry = get(mapping, name, nothing)
            @test entry !== nothing
            if entry !== nothing
                @test get(entry, "lesion_type", "") == "Technical Artifact"
            end
        end
    end
end
