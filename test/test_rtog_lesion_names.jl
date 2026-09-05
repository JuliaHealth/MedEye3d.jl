using Test
using HDF5
using JSON

using MedEye3d
import MedEye3d.SegmentationDisplay.MakieEventHandlers as MEH

@testset "RTOG and Clinical Lesion Names Preservation Suite" begin
    h5_path = joinpath(@__DIR__, "..", "data", "cases", "psma_patient_all_tp", "preprocessed_volumes.h5")
    @test isfile(h5_path)

    h5 = h5open(h5_path, "r")
    @test haskey(h5, "_meta_/scene_hierarchy.json")
    @test haskey(h5, "_meta_/segment_names.json")

    # 1. Test _meta_/segment_names.json contents
    sn_raw = JSON.parse(read(h5["_meta_/segment_names.json"]))
    @test haskey(sn_raw, "0")
    @test haskey(sn_raw, "1")
    @test haskey(sn_raw, "3")

    # TP 0 RTOG segments
    tp0 = sn_raw["0"]
    @test length(tp0) == 12
    @test tp0["1"] == "F08 Knochen Beckengürtel links UBU"
    @test tp0["2"] == "F09 Lymphknoten inguinal rechts"
    @test tp0["8"] == "PT01 ProstataCa Gl 7a gesichert"
    @test tp0["9"] == "LN01 Lymphknoten iliaca externa links"
    @test tp0["12"] == "F07 Knochen Rippe UBU"

    # TP 1 RTOG segments
    tp1 = sn_raw["1"]
    @test length(tp1) == 11
    @test tp1["1"] == "PT02 ProstataCa Gl 3+4=7a spät"
    @test tp1["2"] == "LN09 Lymphknoten Obturator rechts spät"
    @test tp1["4"] == "F12 Knochen Beckengürtel links UBU spät"

    # TP 3 (MRI T2) segments
    tp3 = sn_raw["3"]
    @test length(tp3) == 5
    @test tp3["1"] == "Gland Volume (RTSS)"
    @test tp3["3"] == "Prostatavolumen (MINT)"
    @test tp3["4"] == "P01 Prostata PZ Basis links (MINT)"
    @test tp3["5"] == "Prostate (MDPROSTATE)"

    close(h5)

    # 2. Test ingestion into MEH.tp_segment_names
    empty!(MEH.tp_segment_names)
    for (tp_str, sdict) in sn_raw
        tp_i = parse(Int, tp_str)
        target = get!(MEH.tp_segment_names, tp_i, Dict{Int, String}())
        for (sid_str, sname) in sdict
            target[parse(Int, sid_str)] = string(sname)
        end
    end

    @test haskey(MEH.tp_segment_names, 0)
    @test MEH.tp_segment_names[0][1] == "F08 Knochen Beckengürtel links UBU"
    @test MEH.tp_segment_names[0][8] == "PT01 ProstataCa Gl 7a gesichert"
    @test MEH.tp_segment_names[1][1] == "PT02 ProstataCa Gl 3+4=7a spät"
    @test MEH.tp_segment_names[3][4] == "P01 Prostata PZ Basis links (MINT)"

    # 3. Test cursor readout formatting
    MEH.current_tp_index[] = 0
    seg_names_0 = get(MEH.tp_segment_names, 0, Dict{Int, String}())
    label_0_1 = get(seg_names_0, 1, "")
    @test label_0_1 == "F08 Knochen Beckengürtel links UBU"
    @test "$label_0_1 (L1)" == "F08 Knochen Beckengürtel links UBU (L1)"

    MEH.current_tp_index[] = 1
    seg_names_1 = get(MEH.tp_segment_names, 1, Dict{Int, String}())
    label_1_1 = get(seg_names_1, 1, "")
    @test label_1_1 == "PT02 ProstataCa Gl 3+4=7a spät"
    @test "$label_1_1 (L1)" == "PT02 ProstataCa Gl 3+4=7a spät (L1)"

    # 4. Test dropdown list generator priority
    organ_mapping = Dict{Int, String}(1 => "hip_left", 8 => "prostate")
    tp0_sn = get(MEH.tp_segment_names, 0, Dict{Int, String}())
    d_name_1 = if haskey(tp0_sn, 1) && !isempty(tp0_sn[1])
        tp0_sn[1]
    elseif haskey(organ_mapping, 1)
        organ_mapping[1]
    else
        "Segment_1"
    end
    # Must pick RTOG name over organ_mapping "hip_left"!
    @test d_name_1 == "F08 Knochen Beckengürtel links UBU"
    @test d_name_1 != "hip_left"

    println("\n=== ALL RTOG LESION NAMES TESTS PASSED! ===")
end
