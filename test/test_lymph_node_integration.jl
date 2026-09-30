using Test
using JSON
using HDF5

# Include LesionAssociation directly
include(joinpath(@__DIR__, "..", "src", "display", "LesionAssociation.jl"))
using .LesionAssociation

@testset "Lymph Node Structure Detection" begin
    # Test clinical stations
    test_lns = [
        "Neck_Level_Ia_Submental",
        "Neck_Level_Ib_Submandibular_Left",
        "Neck_Level_IIa_Upper_Jugular_Right",
        "Neck_Level_III_Middle_Jugular_left",
        "Neck_Level_IV_Lower_Jugular_Left",
        "Neck_Level_Va_Upper_Posterior_Triangle_Right",
        "Neck_Level_VI_Anterior_Cervical",
        "Neck_Level_Xb_Occipital_Left",
        "Neck_Parotid_Left",
        "Neck_Retropharyngeal",
        "Thoracic_Station_2_UpperParatracheal_Left",
        "Thoracic_Station_3A_Prevascular_Right",
        "Thoracic_Station_5_Subaortic_Left",
        "Thoracic_Station_6_Paraaortic",
        "Thoracic_Station_7_Subcarinial",
        "Thoracic_Station_8_Paraoesophageal_Left",
        "Thoracic_Station_Prepericardial_Right",
        "Axillary_Level_I_Left",
        "Axillary_Level_III_Right",
        "Axillary_Rotter_Left",
        "Abdominal_Station_1_Right_Paracardial",
        "Abdominal_Station_9_Celiac",
        "Abdominal_Station_16a1_Aortic_Hiatus",
        "Abdominal_Station_16a2_Upper_Middle_Paraaortic",
        "Abdominal_Mesenteric_Interenteric",
        "Abdominal_Obturator_Left",
        "Abdominal_Obturator_Right",
        "Abdominal_Presacral",
        "Abdominal_Pararectal",
        "Abdominal_Common_Iliac_Left",
        "Abdominal_External_Iliac_Right",
        "Abdominal_Internal_Iliac_Left",
        "Deep_Inguinal_Left",
        "Superficial_Inguinal_Right"
    ]
    
    for s in test_lns
        @test is_lymph_node_structure(s) == true
        @test classify_organ_to_lesion_type(s) == "Lymph Node Meta"
    end
    
    # Test non-LN structures
    non_lns = [
        "femur_left", "vertebrae_L1", "rib_left_5", "sacrum", "mandible",
        "liver", "spleen", "kidney_right", "stomach", "esophagus", "prostate",
        "gluteus_maximus_left", "psoas_major_right", "sartorius_left",
        "aorta", "iliac_artery_left", "inferior_vena_cava"
    ]
    for s in non_lns
        @test is_lymph_node_structure(s) == false
    end
end

@testset "Pelvic vs Distant Lymph Node Classification" begin
    # Regional Pelvic (miN1)
    pelvic = [
        "Abdominal_Obturator_Left",
        "Abdominal_Obturator_Right",
        "Abdominal_Internal_Iliac_Left",
        "Abdominal_External_Iliac_Right",
        "Abdominal_Presacral",
        "Abdominal_Pararectal"
    ]
    for s in pelvic
        @test is_pelvic_lymph_node(s) == true
        loc, _ = classify_lymph_node_location(s)
        @test loc == "Pelvic Lymph Node"
    end
    
    # Distant Extra-pelvic (miM1a)
    distant = [
        "Abdominal_Common_Iliac_Left",
        "Abdominal_Station_16a1_Aortic_Hiatus",
        "Abdominal_Station_9_Celiac",
        "Abdominal_Mesenteric_Interenteric",
        "Thoracic_Station_4_LowerParatracheal_Left",
        "Axillary_Level_I_Left",
        "Neck_Level_IIa_Upper_Jugular_Left",
        "Deep_Inguinal_Left"
    ]
    for s in distant
        @test is_pelvic_lymph_node(s) == false
        loc, _ = classify_lymph_node_location(s)
        @test loc == "Distant Lymph Node (Common Iliac, Retroperitoneal, Inguinal, Supraclavicular, Axillary)"
    end
end

@testset "Tissue Priority & pick_best_organ" begin
    # 1. Lymph node wins over muscle even if muscle has more voxels
    counts1 = Dict(1 => 150, 2 => 25)
    names1 = Dict(1 => "gluteus_maximus_left", 2 => "Abdominal_Obturator_Left")
    @test pick_best_organ(counts1, names1) == "Abdominal_Obturator_Left"
    
    # 2. Lymph node wins over adjacent organ even if organ has more voxels
    counts2 = Dict(1 => 80, 2 => 20)
    names2 = Dict(1 => "esophagus", 2 => "Thoracic_Station_8_Paraoesophageal_Left")
    @test pick_best_organ(counts2, names2) == "Thoracic_Station_8_Paraoesophageal_Left"
    
    # 3. Lymph node wins over vessel
    counts3 = Dict(1 => 100, 2 => 30)
    names3 = Dict(1 => "iliac_artery_left", 2 => "Abdominal_Internal_Iliac_Left")
    @test pick_best_organ(counts3, names3) == "Abdominal_Internal_Iliac_Left"
    
    # 4. Tie between two lymph node stations broken by voxel count
    counts4 = Dict(1 => 40, 2 => 80)
    names4 = Dict(1 => "Abdominal_External_Iliac_Left", 2 => "Abdominal_Obturator_Left")
    @test pick_best_organ(counts4, names4) == "Abdominal_Obturator_Left"
    
    # 5. Pure bone lesion picks bone
    counts5 = Dict(1 => 200)
    names5 = Dict(1 => "vertebrae_T12")
    @test pick_best_organ(counts5, names5) == "vertebrae_T12"
    @test classify_organ_to_lesion_type("vertebrae_T12") == "Bone Meta"
    
    # 6. Prostate primary picks prostate
    counts6 = Dict(1 => 300, 2 => 10)
    names6 = Dict(1 => "prostate", 2 => "urinary_bladder")
    @test pick_best_organ(counts6, names6) == "prostate"
    @test classify_organ_to_lesion_type("prostate") == "Prostate"
end

@testset "format_clinical_station_name" begin
    # Pelvic / Abdominal non-station
    @test format_clinical_station_name("Abdominal_Obturator_Left") == "Obturator Lymph Node (Left)"
    @test format_clinical_station_name("Abdominal_Obturator_Right") == "Obturator Lymph Node (Right)"
    @test format_clinical_station_name("Abdominal_Internal_Iliac_Left") == "Internal Iliac Lymph Node (Left)"
    @test format_clinical_station_name("Abdominal_External_Iliac_Right") == "External Iliac Lymph Node (Right)"
    @test format_clinical_station_name("Abdominal_Common_Iliac_Left") == "Common Iliac Lymph Node (Left)"
    @test format_clinical_station_name("Abdominal_Presacral") == "Presacral Lymph Node"
    @test format_clinical_station_name("Abdominal_Pararectal") == "Pararectal Lymph Node"
    @test format_clinical_station_name("Abdominal_Mesenteric_Interenteric") == "Mesenteric Interenteric Lymph Node"
    @test format_clinical_station_name("Abdominal_Renal_Hilar_Left") == "Renal Hilar Lymph Node (Left)"
    
    # JGCA gastric stations
    @test format_clinical_station_name("Abdominal_Station_10_Splenic_Hilum") == "Station 10 Splenic Hilum Lymph Node"
    @test format_clinical_station_name("Abdominal_Station_16a1_Aortic_Hiatus") == "Station 16a1 Aortic Hiatus Lymph Node"
    @test format_clinical_station_name("Abdominal_Station_16b2_Caudal_Paraaortic") == "Station 16b2 Caudal Paraaortic Lymph Node"
    @test format_clinical_station_name("Abdominal_Station_1_Right_Paracardial") == "Station 1 Right Paracardial Lymph Node"
    @test format_clinical_station_name("Abdominal_Station_7_Left_Gastric") == "Station 7 Left Gastric Lymph Node"
    @test format_clinical_station_name("Abdominal_Station_9_Celiac") == "Station 9 Celiac Lymph Node"
    @test format_clinical_station_name("Abdominal_Station_Inferior_Pancreatic") == "Inferior Pancreatic Lymph Node"
    
    # Inguinal
    @test format_clinical_station_name("Deep_Inguinal_Left") == "Deep Inguinal Lymph Node (Left)"
    @test format_clinical_station_name("Superficial_Inguinal_Right") == "Superficial Inguinal Lymph Node (Right)"
    
    # Axillary
    @test format_clinical_station_name("Axillary_Level_I_Left") == "Axillary Level I Lymph Node (Left)"
    @test format_clinical_station_name("Axillary_Level_III_Right") == "Axillary Level III Lymph Node (Right)"
    @test format_clinical_station_name("Axillary_Rotter_Left") == "Rotter Interpectoral Lymph Node (Left)"
    
    # Thoracic / Mediastinal
    @test format_clinical_station_name("Thoracic_Station_2_UpperParatracheal_Left") == "Station 2L Upper Paratracheal Lymph Node"
    @test format_clinical_station_name("Thoracic_Station_3A_Prevascular_Right") == "Station 3A Prevascular Lymph Node (Right)"
    @test format_clinical_station_name("Thoracic_Station_4_LowerParatracheal_Left") == "Station 4L Lower Paratracheal Lymph Node"
    @test format_clinical_station_name("Thoracic_Station_5_Subaortic_Left") == "Station 5 Subaortic (AP Window) Lymph Node"
    @test format_clinical_station_name("Thoracic_Station_7_Subcarinial") == "Station 7 Subcarinal Lymph Node"
    @test format_clinical_station_name("Thoracic_Station_8_Paraoesophageal_Left") == "Station 8 Paraesophageal Lymph Node (Left)"
    @test format_clinical_station_name("Thoracic_Station_Hilar_Interlobar_left") == "Station 10/11 Hilar / Interlobar Lymph Node (Left)"
    @test format_clinical_station_name("Thoracic_Station_Prepericardial_Right") == "Prepericardial Lymph Node (Right)"
    @test format_clinical_station_name("Thoracic_Mammary_Left") == "Internal Mammary Lymph Node (Left)"
    
    # Neck levels
    @test format_clinical_station_name("Neck_Level_Ia_Submental") == "Neck Level Ia Submental Lymph Node"
    @test format_clinical_station_name("Neck_Level_Ib_Submandibular_Left") == "Neck Level Ib Submandibular Lymph Node (Left)"
    @test format_clinical_station_name("Neck_Level_IIa_Upper_Jugular_Left") == "Neck Level IIa Upper Jugular Lymph Node (Left)"
    @test format_clinical_station_name("Neck_Level_IIb_Upper_Jugular_Right") == "Neck Level IIb Upper Jugular Lymph Node (Right)"
    @test format_clinical_station_name("Neck_Level_III_Middle_Jugular_left") == "Neck Level III Middle Jugular Lymph Node (Left)"
    @test format_clinical_station_name("Neck_Level_IV_Lower_Jugular_Right") == "Neck Level IV Lower Jugular Lymph Node (Right)"
    @test format_clinical_station_name("Neck_Level_Va_Upper_Posterior_Triangle_Left") == "Neck Level Va Upper Posterior Triangle Lymph Node (Left)"
    @test format_clinical_station_name("Neck_Level_VI_Anterior_Cervical") == "Neck Level VI Anterior Cervical Lymph Node"
    @test format_clinical_station_name("Neck_Level_Xb_Occipital_Left") == "Neck Level Xb Occipital Lymph Node (Left)"
    @test format_clinical_station_name("Neck_Parotid_Left") == "Parotid Lymph Node (Left)"
    @test format_clinical_station_name("Neck_Retropharyngeal") == "Retropharyngeal Lymph Node"
    
    # Empty / edge cases
    @test format_clinical_station_name("") == ""
end

@testset "format_adjacent_structure_name" begin
    @test format_adjacent_structure_name("iliopsoas_left") == "Iliopsoas Muscle (Left)"
    @test format_adjacent_structure_name("obturator_internus_right") == "Obturator Internus Muscle (Right)"
    @test format_adjacent_structure_name("prostate") == "Prostate Gland"
    @test format_adjacent_structure_name("urinary_bladder") == "Urinary Bladder"
    @test format_adjacent_structure_name("esophagus") == "Esophagus"
    @test format_adjacent_structure_name("gluteus_maximus_left") == "Gluteus Maximus Muscle (Left)"
    @test format_adjacent_structure_name("") == ""
    
    # Vessels
    @test occursin("Iliac Artery", format_adjacent_structure_name("iliac_artery_left"))
    @test occursin("Inferior Vena Cava", format_adjacent_structure_name("inferior_vena_cava"))
    
    # Bones
    @test occursin("Femur", format_adjacent_structure_name("femur_left"))
    @test occursin("Sacrum", format_adjacent_structure_name("sacrum"))
    
    # Lymph nodes delegate to format_clinical_station_name
    @test format_adjacent_structure_name("Abdominal_Obturator_Left") == "Obturator Lymph Node (Left)"
end

@testset "generate_detailed_anatomy_rows" begin
    # Test with primary LN + adjacent structures
    counts = Dict(1 => 100, 2 => 80, 3 => 20, 4 => 5)
    names = Dict(1 => "Abdominal_Obturator_Left", 2 => "iliopsoas_left", 3 => "urinary_bladder", 4 => "femur_left")
    result = generate_detailed_anatomy_rows(counts, names; max_rows=4)
    
    @test startswith(result, "Inside / Contained In:Obturator Lymph Node (Left)")
    @test occursin("Adjacent To:", result)
    @test occursin("Iliopsoas Muscle (Left)", result)
    
    # Test with multiple LN stations
    counts2 = Dict(1 => 60, 2 => 30)
    names2 = Dict(1 => "Abdominal_External_Iliac_Left", 2 => "Abdominal_Obturator_Left")
    result2 = generate_detailed_anatomy_rows(counts2, names2; max_rows=4)
    @test startswith(result2, "Inside / Contained In:External Iliac Lymph Node (Left)")
    @test occursin("Adjacent To:Obturator Lymph Node (Left)", result2)
    
    # Test with no LN overlap → returns empty
    counts3 = Dict(1 => 100)
    names3 = Dict(1 => "liver")
    result3 = generate_detailed_anatomy_rows(counts3, names3; max_rows=4)
    @test result3 == ""
    
    # Test with empty counts
    result4 = generate_detailed_anatomy_rows(Dict{Int,Int}(), Dict{Int,String}(); max_rows=4)
    @test result4 == ""
    
    # Test voxel count threshold (< 3 voxels filtered)
    counts5 = Dict(1 => 50, 2 => 2)  # 2 voxels = too few
    names5 = Dict(1 => "Abdominal_Presacral", 2 => "rectum")
    result5 = generate_detailed_anatomy_rows(counts5, names5; max_rows=4)
    @test startswith(result5, "Inside / Contained In:Presacral Lymph Node")
    @test !occursin("Rectum", result5)  # filtered out (< 3 voxels)
end

@testset "max_anatomy_to_ontology.json correctness" begin
    mapping = JSON.parsefile(joinpath(@__DIR__, "..", "assets", "max_anatomy_to_ontology.json"))
    
    # Verify non-LN entries are fixed
    @test mapping["fused_spine"]["lesion_type"] == "Bone Meta"
    @test mapping["portal_vein"]["lesion_type"] == "Organ Meta"
    @test mapping["tissue_fat"]["lesion_type"] == "Technical Artifact"
    @test mapping["thorax_wall"]["lesion_type"] == "Technical Artifact"
    
    # Verify LN entries have formatted detailed names
    @test mapping["abdominal_obturator_left"]["detailed"] == "Obturator Lymph Node (Left)"
    @test mapping["thoracic_station_7_subcarinial"]["detailed"] == "Station 7 Subcarinal Lymph Node"
    @test mapping["neck_level_iia_upper_jugular_left"]["detailed"] == "Neck Level IIa Upper Jugular Lymph Node (Left)"
    @test mapping["deep_inguinal_right"]["detailed"] == "Deep Inguinal Lymph Node (Right)"
    @test mapping["axillary_rotter_left"]["detailed"] == "Rotter Interpectoral Lymph Node (Left)"
    
    # Verify sublocations are not generic "Lymph Node Station"
    for (key, val) in mapping
        if val["lesion_type"] == "Lymph Node Meta"
            subloc = val["anatomical_sublocation"]
            @test subloc != "Lymph Node Station"
        end
    end
end

println("✅ ALL LYMPH NODE INTEGRATION TESTS PASSED (including detailed anatomy)!")
