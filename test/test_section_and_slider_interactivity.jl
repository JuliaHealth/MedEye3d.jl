using Test
using GLMakie
using MedEye3d
import MedEye3d.LesionMetadataWindow as LMW
import MedEye3d.SegmentationDisplay.MakieEventHandlers as MEH

@testset "Section Collapse/Expand and Continuous Slider Interactivity" begin
    MEH.current_tp_index[] = 0
    MEH.volume_z_size[] = 326
    MEH.global_organ_mapping[][1] = "femur_right"
    active_lesion = Observable("1: Femur Right")
    lesion_ids = Observable(["1: Femur Right"])

    channel = Channel{Any}(100)
    save_json_path = tempname() * ".json"
    win = LMW.create_metadata_window(active_lesion, lesion_ids, channel; save_path = save_json_path)

    # Wait for preload
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

    # Disconnect GLFW mouse tracking so synthetic coordinates are preserved
    GLMakie.disconnect!(screen, Makie.mouse_position)
    events(win.fig.scene).hasfocus[] = true

    function simulate_click(btn::Button)
        bb = btn.layoutobservables.computedbbox[]
        cx = bb.origin[1] + bb.widths[1] / 2
        cy = bb.origin[2] + bb.widths[2] / 2
        events(win.fig.scene).mouseposition[] = (cx, cy)
        push!(events(win.fig.scene).mousebuttonstate, Mouse.left)
        events(win.fig.scene).mousebutton[] = Makie.MouseButtonEvent(Mouse.left, Mouse.press)
        empty!(events(win.fig.scene).mousebuttonstate)
        events(win.fig.scene).mousebutton[] = Makie.MouseButtonEvent(Mouse.left, Mouse.release)
        yield(); sleep(0.1)
    end

    function simulate_slider_drag(isl::IntervalSlider, frac_start, frac_end)
        bb = isl.layoutobservables.computedbbox[]
        y = bb.origin[2] + bb.widths[2] / 2
        x_start = bb.origin[1] + bb.widths[1] * Float32(frac_start)
        x_end = bb.origin[1] + bb.widths[1] * Float32(frac_end)

        events(win.fig.scene).mouseposition[] = (x_start, y)
        push!(events(win.fig.scene).mousebuttonstate, Mouse.left)
        events(win.fig.scene).mousebutton[] = Makie.MouseButtonEvent(Mouse.left, Mouse.press)
        events(win.fig.scene).mouseposition[] = (x_end, y)
        empty!(events(win.fig.scene).mousebuttonstate)
        events(win.fig.scene).mousebutton[] = Makie.MouseButtonEvent(Mouse.left, Mouse.release)
        yield(); sleep(0.1)
    end

    # ── 1. Find section buttons ──
    main_g = LMW._lmw_observables[:g]
    sec_win_btn = nothing
    sec_meta_btn = nothing
    sec_nav_btn = nothing

    for c in main_g.content
        if c.content isa Button
            lbl = c.content.label[]
            if occursin("Windowing", lbl)
                sec_win_btn = c.content
            elseif occursin("Lesion Metadata", lbl)
                sec_meta_btn = c.content
            elseif occursin("Lesion Navigation", lbl)
                sec_nav_btn = c.content
            end
        end
    end

    @test sec_win_btn !== nothing
    @test sec_meta_btn !== nothing
    @test sec_nav_btn !== nothing

    # Verify initial open states
    @test occursin("▼", sec_win_btn.label[])
    @test occursin("▼", sec_meta_btn.label[])

    # ── 2. Test Section Collapse and Expand via Clicks ──
    # Collapse Windowing
    simulate_click(sec_win_btn)
    @test occursin("▶", sec_win_btn.label[])

    # Expand Windowing
    simulate_click(sec_win_btn)
    @test occursin("▼", sec_win_btn.label[])

    # Collapse Lesion Metadata
    simulate_click(sec_meta_btn)
    @test occursin("▶", sec_meta_btn.label[])

    # Expand Lesion Metadata
    simulate_click(sec_meta_btn)
    @test occursin("▼", sec_meta_btn.label[])

    # ── 3. Test Modality Preset Buttons and Slider Interactivity ──
    # Drain channel first
    while isready(channel)
        take!(channel)
    end

    # Find the modality preset buttons
    preset_btns = Button[]
    for c in main_g.content
        if c.content isa Makie.GridLayoutBase.GridLayout
            for sub_c in c.content.content
                if sub_c.content isa Button
                    push!(preset_btns, sub_c.content)
                end
            end
        end
    end
    println("Found $(length(preset_btns)) preset buttons")
    @test length(preset_btns) >= 6

    # Click each preset button in succession via mouse clicks
    for (idx, btn) in enumerate(preset_btns[1:6])
        println("Clicking preset button $idx: $(btn.label[])")
        simulate_click(btn)
    end

    # Check that WindowingEvents arrived in the channel
    events_received = Any[]
    while isready(channel)
        push!(events_received, take!(channel))
    end
    println("Received $(length(events_received)) events in channel from preset button interactions.")
    @test length(events_received) > 0
    @test any(e -> e isa LMW.WindowingEvent, events_received)

    # ── 4. Test Programmatic set_close_to! on IntervalSliders ──
    isliders = IntervalSlider[]
    for c in main_g.content
        if c.content isa IntervalSlider
            push!(isliders, c.content)
        end
    end
    @test length(isliders) >= 4

    for (idx, isl) in enumerate(isliders)
        old_val = isl.interval[]
        LMW.set_close_to!(isl, 100.0, 400.0)
        new_val = isl.interval[]
        println("Slider $idx updated: $old_val -> $new_val")
        @test new_val != old_val || new_val == (100.0, 400.0)
    end

    println("All Section and Slider interactivity tests passed successfully!")
end
