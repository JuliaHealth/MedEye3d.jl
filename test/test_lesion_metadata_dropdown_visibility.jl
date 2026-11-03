using Test
using GLMakie
using MedEye3d
import MedEye3d.LesionMetadataWindow as LMW

@testset "Lesion Metadata Dropdowns Visibility and Options Display" begin
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
    
    # Test all metadata dropdowns in the schema
    dropdown_names = [
        "Lesion Orientation",
        "Anatomic Location",
        "Anatomical Sublocation",
        "Inner Texture / Density / Attenuation",
        "Border and Margin",
        "Lesion Shape",
        "Macroscopic Pattern"
    ]

    for name in dropdown_names
        @test haskey(field_widgets, name)
        w = field_widgets[name]
        @test w isa Makie.Menu
        
        # Verify blockscene exists and has popup child scene
        @test hasproperty(w, :blockscene)
        @test !isempty(w.blockscene.children)
        
        s = w.blockscene.children[1]
        
        # When closed: child scene should be not visible, but plots MUST be visible=true
        @test s.visible[] == false
        for (i, p) in enumerate(s.plots)
            @test p.visible[] == true
        end
        
        # When opened: child scene becomes visible AND all plots remain visible=true
        w.is_open[] = true
        yield(); sleep(0.05)
        
        @test w.is_open[] == true
        @test s.visible[] == true
        for (i, p) in enumerate(s.plots)
            @test p.visible[] == true
        end
        
        # Close menu
        w.is_open[] = false
        yield(); sleep(0.05)
        @test w.is_open[] == false
        @test s.visible[] == false
    end

    # Test toggling section collapse and expand, then verifying dropdown plot visibility
    main_g = LMW._lmw_observables[:g]
    sec_meta_btn = nothing
    for c in main_g.content
        if c.content isa Button && occursin("Lesion Metadata", c.content.label[])
            sec_meta_btn = c.content
            break
        end
    end
    @test sec_meta_btn !== nothing

    # Collapse section
    sec_meta_btn.clicks[] = sec_meta_btn.clicks[] + 1
    yield(); sleep(0.1)

    # Expand section
    sec_meta_btn.clicks[] = sec_meta_btn.clicks[] + 1
    yield(); sleep(0.1)

    # After collapse and expand, opening a dropdown MUST have plots visible=true
    orient_w = field_widgets["Lesion Orientation"]
    orient_s = orient_w.blockscene.children[1]
    
    orient_w.is_open[] = true
    yield(); sleep(0.05)
    @test orient_w.is_open[] == true
    @test orient_s.visible[] == true
    for p in orient_s.plots
        @test p.visible[] == true
    end
    orient_w.is_open[] = false

    println("All Lesion Metadata Dropdowns Visibility tests passed successfully!")
end
