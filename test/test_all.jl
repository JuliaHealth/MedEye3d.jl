using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Test
using MedEye3d.LesionAssociation
using MedEye3d.InferenceClient
using MedEye3d.HeuristicsEngine
using MedImages
using MedEye3d.LesionMetadataWindow
using JSON

@testset "MedEye3d Full Test Suite" begin

    @testset "LesionAssociation Module" begin
        # Reset the static map
        empty!(LesionAssociation.OVERLAP_MAPPING)
        
        @test isempty(LesionAssociation.OVERLAP_MAPPING)
        
        # Test map_link
        LesionAssociation.map_link("time1", "time2", "1")
        
        @test length(LesionAssociation.OVERLAP_MAPPING) >= 1
        
        # Test get_children
        children = LesionAssociation.get_children("time1", "time2", "1")
        
        @test "1" in children
    end
    
    @testset "InferenceClient Lifecycle & Model Prompt" begin
        # Test function definitions
        @test isdefined(InferenceClient, :run_helpnet_inference)
        @test isdefined(InferenceClient, :run_nninteractive)
        @test isdefined(InferenceClient, :prompt_start_ai_models)
        @test isdefined(InferenceClient, :is_worker_reachable)
        @test isdefined(InferenceClient, :is_ai_enabled)
        @test isdefined(InferenceClient, :set_ai_enabled!)

        # Test enable/disable toggle
        InferenceClient.set_ai_enabled!(false)
        @test InferenceClient.is_ai_enabled() == false
        @test ENV["MEDEYE3D_START_AI"] == "0"
        
        InferenceClient.set_ai_enabled!(true)
        @test InferenceClient.is_ai_enabled() == true
        @test ENV["MEDEYE3D_START_AI"] == "1"

        # Test CLI flags override
        @test InferenceClient.prompt_start_ai_models(["--no-ai"]) == false
        @test InferenceClient.is_ai_enabled() == false

        @test InferenceClient.prompt_start_ai_models(["--ai"]) == true
        @test InferenceClient.is_ai_enabled() == true

        @test InferenceClient.prompt_start_ai_models(["--disable-ai"]) == false
        @test InferenceClient.prompt_start_ai_models(["--enable-ai"]) == true

        # Test environment variable override
        ENV["MEDEYE3D_START_AI"] = "0"
        @test InferenceClient.prompt_start_ai_models(String[]) == false
        @test InferenceClient.is_ai_enabled() == false

        ENV["MEDEYE3D_START_AI"] = "1"
        @test InferenceClient.prompt_start_ai_models(String[]) == true
        @test InferenceClient.is_ai_enabled() == true

        delete!(ENV, "MEDEYE3D_START_AI")

        # Test reachability check on non-existent port returns false without throwing
        @test InferenceClient.is_worker_reachable(port=59999) == false

        # Test graceful handling of disabled AI in inference calls
        InferenceClient.set_ai_enabled!(false)
        dummy_vol = zeros(Float32, 10, 10, 10)
        @test InferenceClient.run_helpnet_inference(dummy_vol, dummy_vol, nothing, 5, 5, 5) === nothing
        @test InferenceClient.run_nninteractive(dummy_vol, dummy_vol, [[0, 0, 0]], 5, 5, 5) === nothing
        @test InferenceClient.run_bone_subsegmentation_remote(dummy_vol, dummy_vol, (1.0, 1.0, 1.0)) == (nothing, nothing)
        @test InferenceClient.preload_ct_for_nninteractive(dummy_vol) === nothing
    end
    
    @testset "LesionMetadataWindow Schema and Serialization" begin
        # Test schema loading from assets/def.json
        schema = LesionMetadataWindow.load_schema()
        @test length(schema) == 20
        @test schema[1].short == "Radioligand Type"
        
        # Test that dropdown options are richly populated (not empty or stub)
        schema_dict = Dict(q.short => q for q in schema)
        @test haskey(schema_dict, "Anatomic Location")
        @test length(schema_dict["Anatomic Location"].options) == 20
        
        @test haskey(schema_dict, "Anatomical Sublocation")
        @test length(schema_dict["Anatomical Sublocation"].options) == 45
        
        @test haskey(schema_dict, "Alternative Hypothesis (False Positive)")
        @test length(schema_dict["Alternative Hypothesis (False Positive)"].options) == 85
        
        @test haskey(schema_dict, "Inner Texture / Density / Attenuation")
        @test length(schema_dict["Inner Texture / Density / Attenuation"].options) == 18

        # Test anatomy mapping (assets/max_anatomy_to_ontology.json)
        anatomy_mapping = LesionMetadataWindow.load_anatomy_mapping()
        @test length(anatomy_mapping) >= 200
        @test haskey(anatomy_mapping, "femur_left") || haskey(anatomy_mapping, "liver")

        # Test RadLex and Foundational Anatomy ontologies
        radlex_terms = LesionMetadataWindow.load_radlex()
        @test length(radlex_terms) > 100
        @test radlex_terms != ["(none)"]

        foundational_terms = LesionMetadataWindow.load_anatomy_ontology()
        @test length(foundational_terms) > 100
        @test foundational_terms != ["(none)"]

        # Test custom options baseline
        custom_opts = LesionMetadataWindow.load_custom_options()
        @test isa(custom_opts, Dict)
        
        # Test saving and loading annotations
        temp_path = tempname() * ".json"
        
        # Create some mock annotations
        mock_data = Dict(
            "lesion_1" => Dict("Radioligand Type" => "18F-PSMA-1007", "Anatomic Location" => "Prostate Gland"),
            "lesion_2" => Dict("Radioligand Type" => "68Ga-PSMA-11", "Lesion Shape" => "Round")
        )
        
        # Write to temp file
        open(temp_path, "w") do f
            JSON.print(f, mock_data, 2)
        end
        
        # Load it using the module's function
        loaded_data = LesionMetadataWindow.load_annotations(temp_path)
        @test haskey(loaded_data, "lesion_1")
        @test loaded_data["lesion_1"]["Anatomic Location"] == "Prostate Gland"
        
        # Test saving it back
        temp_path2 = tempname() * ".json"
        LesionMetadataWindow.save_annotations(loaded_data, temp_path2)
        
        @test isfile(temp_path2)
        reloaded_data = JSON.parse(read(temp_path2, String))
        @test reloaded_data == loaded_data
        
        # Cleanup
        rm(temp_path)
        rm(temp_path2)
    end

    @testset "HeuristicsEngine Inference" begin
        using Dates
        
        # Create a mock CT image
        mock_ct = MedImages.MedImage(
            voxel_data = fill(1000.0f0, 10, 10, 10),
            spacing = (1.0, 1.0, 1.0),
            origin = (0.0, 0.0, 0.0),
            direction = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
            image_type = MedImages.MedImage_data_struct.CT_type,
            image_subtype = MedImages.MedImage_data_struct.CT_subtype,
            date_of_saving = Dates.now(),
            acquistion_time = Dates.now(),
            patient_id = "test"
        )
        
        mock_pet = mock_ct
        
        # Create a mock mask
        mock_mask = zeros(UInt8, 10, 10, 10)
        mock_mask[4:6, 4:6, 4:6] .= 1
        
        heuristics = HeuristicsEngine.compute_heuristics(mock_ct, mock_pet, mock_mask)
        @test haskey(heuristics, "Inner Texture / Density / Attenuation")
        @test heuristics["Inner Texture / Density / Attenuation"] == "Sclerotic / Blastic"
        @test haskey(heuristics, "Lesion Shape")
    end
end
