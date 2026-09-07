#!/usr/bin/env julia
# Test: Simulate the paint → organ mapping → auto-fill chain
# This tests the full code path that should fire when painting stops.
# Run: julia --project test/test_paint_autofill_chain.jl

using Test
using JSON
using HDF5

@testset "Paint → Organ Mapping → Auto-fill Chain" begin
    mapping = JSON.parsefile("assets/max_anatomy_to_ontology.json")
    
    h5 = h5open("data/cases/psma_patient_all_tp/preprocessed_volumes.h5", "r")
    ts_raw = JSON.parse(read(h5["_meta_/max_anatomy_labels.json"]))
    ts_names = Dict{Int,String}(parse(Int, k) => v for (k, v) in ts_raw)
    atlas = read(h5["ATLAS/max_anatomy"])
    mask = read(h5["BASELINE/PET_Lesions_0.nii.gz"])
    close(h5)
    
    @testset "Step 1: Atlas and mask dimensions match" begin
        @test size(atlas) == size(mask)
        println("  Atlas: $(size(atlas)), Mask: $(size(mask))")
    end
    
    @testset "Step 2: classify_and_pick_best_organ works" begin
        # Include LesionAssociation module
        include(joinpath(@__DIR__, "..", "src", "display", "LesionAssociation.jl"))
        
        # Find which lesion IDs exist in the mask
        unique_ids = sort(unique(mask))
        lesion_ids = filter(x -> x > 0, unique_ids)
        println("  Found $(length(lesion_ids)) lesion IDs in mask: $lesion_ids")
        
        for lid in lesion_ids
            lid_int = Int(lid)
            organ_name = LesionAssociation.classify_and_pick_best_organ(mask, atlas, ts_names, lid_int)
            if !isempty(organ_name)
                @test !isempty(organ_name)
                println("  Lesion $lid_int → organ: '$organ_name'")
                
                # Verify ontology lookup
                key = lowercase(strip(organ_name))
                entry = get(mapping, key, get(mapping, organ_name, nothing))
                if entry !== nothing
                    det = get(entry, "detailed", "?")
                    side = get(entry, "side", "")
                    lt = get(entry, "lesion_type", "?")
                    println("    → Ontology: detailed='$det', side='$side', lesion_type='$lt'")
                    @test !isempty(det)
                    @test lt in ("Bone Meta", "Organ Meta", "Prostate", "Technical Artifact")
                else
                    println("    ⚠ No ontology entry for '$organ_name'")
                    @test false  # Every TS organ should have an ontology entry
                end
            else
                println("  Lesion $lid_int → NO organ found (zero atlas overlap)")
            end
        end
    end
    
    @testset "Step 3: lookup_anatomy function works for all TS names" begin
        # Test the function that the listener uses
        function test_lookup_anatomy(raw_organ)
            key = lowercase(strip(raw_organ))
            return get(mapping, key, get(mapping, raw_organ, nothing))
        end
        
        # Test all TS names
        failed = String[]
        for (id, name) in ts_names
            entry = test_lookup_anatomy(name)
            if entry === nothing
                push!(failed, "$id: $name")
            end
        end
        @test isempty(failed)
        if !isempty(failed)
            println("  FAILED lookups:")
            for f in failed; println("    $f"); end
        end
    end
    
    @testset "Step 4: Side NA filtering works" begin
        for side_val in ["NA", "N/A", "na", "Na"]
            _side = uppercase(side_val) in ("NA", "N/A") ? "" : side_val
            @test isempty(_side)
        end
        for side_val in ["Right", "Left", ""]
            _side = uppercase(side_val) in ("NA", "N/A") ? "" : side_val
            @test _side == side_val
        end
    end
end
