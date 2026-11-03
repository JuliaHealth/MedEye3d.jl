using Test
using GLMakie
using MedEye3d
import MedEye3d.LesionMetadataWindow as LMW

@testset "Lesion Metadata Combined Rows & Visibility" begin
    active_lesion = Observable("1: Femur Right")
    lesion_ids = Observable(["1: Femur Right"])
    channel = Channel{Any}(100)
    save_json_path = tempname() * ".json"
    
    win = LMW.create_metadata_window(active_lesion, lesion_ids, channel; save_path = save_json_path)

    t0 = time()
    while !get(LMW._lmw_observables, :gui_preload_done, Ref(false))[] && time() - t0 < 10.0
        sleep(0.1)
    end
    sleep(0.5)

    screen = LMW.display_metadata_window(win.fig)
    LMW._lmw_observables[:loading_overlay_bg].visible = false
    LMW._lmw_observables[:loading_overlay_txt].visible = false
    LMW._login_authenticated[] = true
    LMW._lmw_observables[:login_visible][] = false

    field_widgets = LMW._lmw_observables[:field_widgets]
    q_row_indices = LMW._lmw_observables[:q_row_indices]
    row_visibility = LMW._lmw_observables[:row_visibility_state]
    
    # 1. Verify SUV max, SUV mean, and PSMA-RADS 2.0 are textboxes
    @test haskey(field_widgets, "SUV max")
    @test haskey(field_widgets, "SUV mean")
    @test haskey(field_widgets, "PSMA-RADS 2.0")
    @test haskey(field_widgets, "PRIMARY score pattern?")
    
    tb_max = field_widgets["SUV max"]
    tb_mean = field_widgets["SUV mean"]
    tb_rads = field_widgets["PSMA-RADS 2.0"]
    tb_primary = field_widgets["PRIMARY score pattern?"]
    
    @test tb_max isa Makie.Textbox
    @test tb_mean isa Makie.Textbox
    @test tb_rads isa Makie.Textbox
    @test tb_primary isa Makie.Textbox
    
    # 2. Verify SUV max, SUV mean, and PSMA-RADS share the EXACT same row
    suv_max_rows = q_row_indices["SUV max"]
    suv_mean_rows = q_row_indices["SUV mean"]
    rads_rows = q_row_indices["PSMA-RADS 2.0"]
    primary_rows = q_row_indices["PRIMARY score pattern?"]
    
    @test suv_max_rows == suv_mean_rows
    @test suv_max_rows == rads_rows
    
    # 3. Verify PRIMARY score pattern has its OWN row (not shared with SUV or RADS)
    @test primary_rows != suv_max_rows
    
    # 4. Verify No CT Correlate toggle is present and registered
    @test haskey(LMW._lmw_observables, :no_ct_toggle)
    no_ct = LMW._lmw_observables[:no_ct_toggle]
    @test no_ct isa Makie.Toggle
    @test no_ct.active[] == false
    
    # 5. Test type button changes and dynamic visibility
    btn_prostate = LMW._lmw_observables[:btn_type_prostate]
    btn_bone = LMW._lmw_observables[:btn_type_bone]
    
    # Switch to Prostate: PRIMARY row should become visible
    btn_prostate.clicks[] = btn_prostate.clicks[] + 1
    yield(); sleep(0.2)
    @test LMW._lmw_observables[:active_lesion_type][] == "Prostate"
    primary_row = first(primary_rows)
    @test get(row_visibility, primary_row, false) == true
    
    # Switch to Bone Meta: PRIMARY row should become hidden
    btn_bone.clicks[] = btn_bone.clicks[] + 1
    yield(); sleep(0.2)
    @test LMW._lmw_observables[:active_lesion_type][] == "Bone Meta"
    @test get(row_visibility, primary_row, true) == false
    
    # SUV row must remain visible regardless of type
    suv_row = first(suv_max_rows)
    @test get(row_visibility, suv_row, false) == true
    
    # 6. Test No CT Correlate toggle
    no_ct.active[] = true
    yield(); sleep(0.2)
    
    # Inner Texture is CT-specific and should now be hidden
    it_rows = q_row_indices["Inner Texture / Density / Attenuation"]
    it_row = first(it_rows)
    @test get(row_visibility, it_row, true) == false
    
    # Reset No CT Correlate toggle
    no_ct.active[] = false
    yield(); sleep(0.2)
    @test get(row_visibility, it_row, false) == true
    
    println("All Combined Rows & Visibility tests passed successfully!")
end
