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
    
    @testset "InferenceClient JSON Serialization" begin
        # We can't guarantee a Python backend in the automated test, 
        # so we will just test that the functions are accessible.
        @test isdefined(InferenceClient, :run_helpnet_inference)
        @test isdefined(InferenceClient, :run_nninteractive)
    end
    
    @testset "LesionMetadataWindow Schema and Serialization" begin
        # Test schema loading
        schema = LesionMetadataWindow.load_schema()
        @test length(schema) > 0
        @test schema[1].short == "Radioligand Type"
        
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
