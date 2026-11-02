using Test
using GLMakie
using MedEye3d
import MedEye3d.LesionMetadataWindow as LMW
import MedEye3d.SegmentationDisplay.MakieEventHandlers as MEH
import MedEye3d.MakieEvents as ME
import MedEye3d.ForDisplayStructs: MultiImage, QuadImage

@testset "Multi-Panel TimePoint Dropdowns and Compare Mode" begin
    MEH.current_tp_index[] = 0
    MEH.compare_right_tp[] = -1
    MEH.compare_mode[] = false
    empty!(MEH.tp_labels)
    MEH.tp_labels[0] = "Baseline TP0"
    MEH.tp_labels[1] = "Follow-up TP1"
    
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

    menu_single = LMW._lmw_observables[:menu_tp_single]
    menu_left   = LMW._lmw_observables[:menu_tp_left]
    menu_right  = LMW._lmw_observables[:menu_tp_right]

    @test menu_single isa Menu
    @test menu_left isa Menu
    @test menu_right isa Menu

    # Drain channel of initialization events
    while isready(channel); take!(channel); end

    # Trigger tp_switched to sync menus with tp_labels
    MEH.tp_switched[] = MEH.tp_switched[] + 1
    yield(); sleep(0.1)

    @test length(menu_single.options[]) == 2
    @test length(menu_left.options[]) == 2
    @test length(menu_right.options[]) == 2

    # Left should be TP 0, Right default should be TP 1
    @test startswith(string(menu_single.selection[]), "0:")
    @test startswith(string(menu_left.selection[]), "0:")
    @test startswith(string(menu_right.selection[]), "1:")

    # Find sec_view button
    main_g = LMW._lmw_observables[:g]
    sec_view_btn = nothing
    for c in main_g.content
        if c.content isa Button && occursin("Viewport & Multi-Planar Layout", c.content.label[])
            sec_view_btn = c.content
            break
        end
    end
    @test sec_view_btn !== nothing

    # Expand section "Viewport & Multi-Planar Layout"
    sec_view_btn.clicks[] = sec_view_btn.clicks[] + 1
    yield(); sleep(0.2)

    # In single-TP mode, menu_single must be visible and menu_left/menu_right must be hidden
    @test menu_single.blockscene.visible[] == true
    @test menu_left.blockscene.visible[] == false
    @test menu_right.blockscene.visible[] == false

    # Turn ON compare mode
    LMW._lmw_observables[:obs_set_compare_mode][] = true
    yield(); sleep(0.2)

    # Drain and check channel for CompareTimePointsEvent(true)
    cmp_event_found = false
    while isready(channel)
        ev = take!(channel)
        if ev isa ME.CompareTimePointsEvent && ev.compare == true
            cmp_event_found = true
        end
    end
    @test cmp_event_found == true

    # In compare mode, menu_single must be hidden and menu_left/menu_right must be visible
    @test menu_single.blockscene.visible[] == false
    @test menu_left.blockscene.visible[] == true
    @test menu_right.blockscene.visible[] == true

    # Collapse and re-expand sec_view to verify visibility is preserved
    sec_view_btn.clicks[] = sec_view_btn.clicks[] + 1
    yield(); sleep(0.1)
    @test menu_left.blockscene.visible[] == false
    @test menu_right.blockscene.visible[] == false

    sec_view_btn.clicks[] = sec_view_btn.clicks[] + 1
    yield(); sleep(0.1)
    # After re-expanding while in compare mode, compare menus should be visible again
    @test menu_single.blockscene.visible[] == false
    @test menu_left.blockscene.visible[] == true
    @test menu_right.blockscene.visible[] == true

    # Test selecting from menu_tp_right
    opts = menu_right.options[]
    idx0 = findfirst(o -> startswith(o, "0:"), opts)
    menu_right.i_selected[] = idx0
    menu_right.selection[] = opts[idx0]
    yield(); sleep(0.1)

    ev_right = nothing
    while isready(channel)
        ev = take!(channel)
        if ev isa ME.SetTimePointEvent
            ev_right = ev
        end
    end
    @test ev_right !== nothing
    @test ev_right.tp_index == 0
    @test ev_right.panel == 5

    # Test selecting from menu_tp_left
    idx1 = findfirst(o -> startswith(o, "1:"), opts)
    menu_left.i_selected[] = idx1
    menu_left.selection[] = opts[idx1]
    yield(); sleep(0.1)

    ev_left = nothing
    while isready(channel)
        ev = take!(channel)
        if ev isa ME.SetTimePointEvent
            ev_left = ev
        end
    end
    @test ev_left !== nothing
    @test ev_left.tp_index == 1
    @test ev_left.panel == 1

    # Turn OFF compare mode
    LMW._lmw_observables[:obs_set_compare_mode][] = false
    yield(); sleep(0.2)

    @test menu_single.blockscene.visible[] == true
    @test menu_left.blockscene.visible[] == false
    @test menu_right.blockscene.visible[] == false

    # Test MakieEventHandlers vertex updates for single-window compare mode
    states = [MEH.StateDataFields() for _ in 1:5]
    dummy_entry = MEH.TpCacheEntry(
        zeros(Float32, 2, 2, 2),
        zeros(Float32, 2, 2, 2),
        zeros(Int8, 2, 2, 2),
        zeros(Int8, 2, 2, 2),
        nothing,
        zeros(Int16, 2, 2, 2),
        nothing
    )
    MEH.tp_data_cache[0] = dummy_entry
    MEH.tp_data_cache[1] = dummy_entry

    # reactToCompareTimePoints(true) in single window
    MEH.reactToCompareTimePoints(ME.CompareTimePointsEvent(true), states)
    @test MEH.compare_mode[] == true
    @test states[1].displayMode == MultiImage && states[1].calcDimsStruct.imagePos == 1
    @test states[5].displayMode == MultiImage && states[5].calcDimsStruct.imagePos == 2
    @test states[2].calcDimsStruct.imagePos == -1
    @test states[3].calcDimsStruct.imagePos == -1
    @test states[4].calcDimsStruct.imagePos == -1

    # reactToCompareTimePoints(false) restores quad
    MEH.reactToCompareTimePoints(ME.CompareTimePointsEvent(false), states)
    @test MEH.compare_mode[] == false
    @test states[1].displayMode == QuadImage && states[1].calcDimsStruct.imagePos == 1
    @test states[3].displayMode == QuadImage && states[3].calcDimsStruct.imagePos == 3
    @test states[4].displayMode == QuadImage && states[4].calcDimsStruct.imagePos == 4
    @test states[5].calcDimsStruct.imagePos == -1

    # Clean up test cache
    empty!(MEH.tp_data_cache)
    println("All Multi-Panel TimePoint Dropdowns and Compare Mode tests passed!")
end
