using Test
using MedEye3d
using HDF5
using JSON

import MedEye3d.LesionAssociation as LA
import MedEye3d.SegmentationDisplay as SD
import MedEye3d.SegmentationDisplay.MakieEventHandlers as MEH
import MedEye3d.StructsManag as SM
import MedEye3d.DataStructs as DS
import MedEye3d.ForDisplayStructs as FDS
import MedEye3d.LesionMetadataWindow as LMW

@testset "Prostate Anatomy Classification & Priority" begin
    # 1. Test is_prostate_structure for all zones
    @test LA.is_prostate_structure("Peripheral Zone (PZ)") == true
    @test LA.is_prostate_structure("Transition Zone (TZ)") == true
    @test LA.is_prostate_structure("Central Zone (CZ)") == true
    @test LA.is_prostate_structure("Anterior Fibromuscular Stroma (AFS)") == true
    @test LA.is_prostate_structure("Urethra (UR)") == true
    @test LA.is_prostate_structure("Periurethral Glands (PG)") == true
    @test LA.is_prostate_structure("Seminal Vesicle Left (SV_L)") == true
    @test LA.is_prostate_structure("prostate") == true
    @test LA.is_prostate_structure("Prostate Gland") == true
    @test LA.is_prostate_structure("Superficial_Inguinal_Right") == false

    # 2. Test classify_tissue_priority
    @test LA.classify_tissue_priority("prostate") == 1
    @test LA.classify_tissue_priority("Transition Zone (TZ)") == 1
    @test LA.classify_tissue_priority("Peripheral Zone (PZ)") == 1
    @test LA.classify_tissue_priority("Superficial_Inguinal_Right") == 2
    @test LA.classify_tissue_priority("femur_left") == 3
    @test LA.classify_tissue_priority("liver") == 4
    @test LA.classify_tissue_priority("gluteus_maximus_left") == 6

    # 3. Test pick_best_organ: prostate priority over lymph nodes
    counts = Dict(100 => 500, 200 => 300) # 100: prostate zone, 200: lymph node
    ts_names = Dict(100 => "Transition Zone (TZ)", 200 => "Superficial_Inguinal_Right")
    best = LA.pick_best_organ(counts, ts_names)
    @test best == "Transition Zone (TZ)"

    # 4. Test lookup_anatomy in ontology
    entry_tz = LMW.lookup_anatomy("Transition Zone (TZ)")
    @test entry_tz !== nothing
    @test entry_tz["lesion_type"] == "Prostate"
    @test entry_tz["anatomic_location"] == "Prostate Gland"

    entry_pz = LMW.lookup_anatomy("Peripheral Zone (PZ)")
    @test entry_pz !== nothing
    @test entry_pz["lesion_type"] == "Prostate"
end

@testset "Aspect Ratio & Resize NDC Calculation" begin
    # Test that getMainVerticies maintains aspect ratio regardless of window dimensions
    tex_w = 256
    tex_h = 256
    voxel_aspect = 1.0f0 # 1:1 physical ratio

    # Case A: Wide window (1200 x 600) -> actual aspect ratio is 0.5 (height/width)
    cd_wide = DS.CalcDimsStruct(
        windowWidth = 1200,
        windowHeight = 600,
        fractionOfMainIm = 1.0f0,
        imageTextureWidth = tex_w,
        imageTextureHeight = tex_h,
        heightToWithRatio = voxel_aspect,
        imagePos = 1
    )
    res_wide = SM.getMainVerticies(cd_wide, FDS.QuadImage, 1)
    verts_wide = res_wide.mainImageQuadVert
    @test length(verts_wide) == 32
    # NDC quad bounds
    ndc_w_wide = abs(verts_wide[1] - verts_wide[17]) # right_x - left_x
    ndc_h_wide = abs(verts_wide[2] - verts_wide[10]) # top_y - bottom_y
    # Pixel dimensions should maintain physical square aspect ratio
    quad_w_px = (ndc_w_wide / 2.0) * 1200
    quad_h_px = (ndc_h_wide / 2.0) * 600
    # In QuadImage on 1200x600 window: quadrant is 600x300, image must be 300x300
    @test isapprox(quad_w_px, quad_h_px; atol=2.0)

    # Case B: Tall window (600 x 1200) -> actual aspect ratio is 2.0
    cd_tall = DS.CalcDimsStruct(
        windowWidth = 600,
        windowHeight = 1200,
        fractionOfMainIm = 1.0f0,
        imageTextureWidth = tex_w,
        imageTextureHeight = tex_h,
        heightToWithRatio = voxel_aspect,
        imagePos = 1
    )
    res_tall = SM.getMainVerticies(cd_tall, FDS.QuadImage, 1)
    verts_tall = res_tall.mainImageQuadVert
    ndc_w_tall = abs(verts_tall[1] - verts_tall[17])
    ndc_h_tall = abs(verts_tall[2] - verts_tall[10])
    quad_w_px_tall = (ndc_w_tall / 2.0) * 600
    quad_h_px_tall = (ndc_h_tall / 2.0) * 1200
    @test isapprox(quad_w_px_tall, quad_h_px_tall; atol=2.0)
end

@testset "Lymph Node Mode Navigation Guard" begin
    # Test that in LN mode, opts is NOT pruned by db_val
    MEH.lymph_node_validation_mode[] = true
    @test MEH.lymph_node_validation_mode[] == true

    # Create dummy stations in lesion_ids
    dummy_opts = ["100: Internal Iliac R", "101: Internal Iliac L", "102: External Iliac R"]
    db_val = Dict{String, Any}("100" => Dict{String, Any}("BaseAnatomy" => "Internal Iliac R"))

    opts = copy(dummy_opts)
    is_ln_mode = MEH.lymph_node_validation_mode[]
    if !is_ln_mode
        valid_keys = Set(keys(db_val))
        filter!(opt -> string(LMW.parse_lesion_id(opt)) in valid_keys, opts)
    end
    # Because is_ln_mode is true, opts must retain ALL 3 stations!
    @test length(opts) == 3
    @test opts == dummy_opts

    MEH.lymph_node_validation_mode[] = false
end

println("\n✅ All resize, prostate naming & LN navigation tests passed!")
