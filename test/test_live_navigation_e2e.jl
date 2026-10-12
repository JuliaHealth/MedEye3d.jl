using Test
using GLMakie
using MedEye3d
import MedEye3d.LesionMetadataWindow as LMW
import MedEye3d.SegmentationDisplay.MakieEventHandlers as MEH

@testset "Live Lesion Navigation and Refresh Anatomy E2E" begin
    # Mock MEH
    MEH.current_tp_index[] = 0
    MEH.volume_z_size[] = 326
    MEH.global_organ_mapping[][1] = "femur_right"
    MEH.global_organ_mapping[][2] = "sartorius_right"
    MEH.global_organ_mapping[][7] = "lymph_node_obturator_right"
    MEH.global_organ_mapping[][8] = "prostate"
    MEH.tp_segment_names[0] = Dict(1 => "femur_right", 2 => "sartorius_right", 7 => "lymph_node_obturator_right", 8 => "prostate")

    active_lesion = Observable("1: Femur Right")
    lesion_ids = Observable(["1: Femur Right", "2: Sartorius Right", "7: Obturator Lymph Node (Right)", "8: Prostate Gland"])

    save_json_path = tempname() * ".json"
    win = LMW.create_metadata_window(active_lesion, lesion_ids, nothing; save_path = save_json_path)

    # Allow async DB init
    sleep(0.5)

    # Initial Lesion 1
    @test occursin("femur", lowercase(string(LMW._lmw_observables[:menu_base_anat].selection[])))
    @test occursin("Appendicular", string(LMW._lmw_observables[:field_widgets]["Anatomic Location"].selection[]))

    # Navigate to Lesion 7
    active_lesion[] = "7: Obturator Lymph Node (Right)"
    sleep(0.5)
    @test occursin("Obturator Lymph Node", string(LMW._lmw_observables[:menu_base_anat].selection[]))
    @test LMW._lmw_observables[:anat_active_count][] > 0

    # Click Refresh Anatomy on Lesion 7
    LMW._lmw_observables[:btn_refresh_anat].clicks[] += 1
    sleep(0.5)
    @test occursin("Obturator Lymph Node", string(LMW._lmw_observables[:menu_base_anat].selection[]))
    @test occursin("Pelvic Lymph Node", string(LMW._lmw_observables[:field_widgets]["Anatomic Location"].selection[]))
    @test LMW._lmw_observables[:anat_active_count][] > 0

    # Navigate to Lesion 8
    active_lesion[] = "8: Prostate Gland"
    sleep(0.5)
    @test occursin("prostate", lowercase(string(LMW._lmw_observables[:menu_base_anat].selection[])))

    # Navigate back to Lesion 7 (verify independence from 8)
    active_lesion[] = "7: Obturator Lymph Node (Right)"
    sleep(0.5)
    @test occursin("Obturator Lymph Node", string(LMW._lmw_observables[:menu_base_anat].selection[]))
    @test occursin("Pelvic Lymph Node", string(LMW._lmw_observables[:field_widgets]["Anatomic Location"].selection[]))

    # Navigate to Lesion 1 (verify independence from 7 and 8)
    active_lesion[] = "1: Femur Right"
    sleep(0.5)
    @test occursin("femur", lowercase(string(LMW._lmw_observables[:menu_base_anat].selection[])))
    @test occursin("Appendicular", string(LMW._lmw_observables[:field_widgets]["Anatomic Location"].selection[]))

    # Verify DB content independence
    db = LMW._lmw_observables[:lesion_db][]
    @test occursin("femur", lowercase(get(db["1"], "BaseAnatomy", "")))
    @test occursin("obturator", lowercase(get(db["7"], "BaseAnatomy", "")))
    @test occursin("prostate", lowercase(get(db["8"], "BaseAnatomy", "")))
    @test get(db["1"], "BaseAnatomy", "") != get(db["7"], "BaseAnatomy", "")
    @test get(db["7"], "BaseAnatomy", "") != get(db["8"], "BaseAnatomy", "")
end
