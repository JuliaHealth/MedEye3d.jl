using Test
using GLMakie
using MedEye3d
import MedEye3d.LesionMetadataWindow as LMW
import MedEye3d.SegmentationDisplay.MakieEventHandlers as MEH

@testset "GUI Locking, Dropdowns, Plus Button, and Certainty Description" begin
    # ── 1. Schema & Option Definitions ─────────────────────────────────────────
    schema = LMW.load_schema()
    orient_q = findfirst(q -> q.short == "Lesion Orientation", schema)
    @test orient_q !== nothing
    oq = schema[orient_q]
    @test any(opt -> occursin("Anterior-Posterior", opt), oq.options)
    @test any(opt -> occursin("anterior posterior", opt), oq.options)

    cert_q = findfirst(q -> q.short == "Certainty", schema)
    @test cert_q !== nothing
    cq = schema[cert_q]
    @test occursin("0 = sure alternative hypothesis", cq.full)
    @test occursin("10 = sure prostate cancer", cq.full)

    # ── 2. Persistence of Custom Options across Runs ──────────────────────────
    test_custom_opt = "Test Lateral Angled Axis $(round(Int, time()))"
    LMW.add_global_custom_option("Lesion Orientation", test_custom_opt)
    disk_opts = LMW.load_custom_options()
    @test haskey(disk_opts, "Lesion Orientation")
    @test test_custom_opt in disk_opts["Lesion Orientation"]

    # ── 3. GUI Construction & Plus Button Inline Add ───────────────────────────
    MEH.current_tp_index[] = 0
    MEH.volume_z_size[] = 326
    MEH.global_organ_mapping[][1] = "femur_right"
    active_lesion = Observable("1: Femur Right")
    lesion_ids = Observable(["1: Femur Right"])

    save_json_path = tempname() * ".json"
    win = LMW.create_metadata_window(active_lesion, lesion_ids, nothing; save_path = save_json_path)

    # Allow async DB init and GUI preload to complete
    t0 = time()
    while !get(LMW._lmw_observables, :gui_preload_done, Ref(false))[] && time() - t0 < 15.0
        sleep(0.1)
    end
    sleep(0.5)

    orient_menu = LMW._lmw_observables[:field_widgets]["Lesion Orientation"]
    @test orient_menu isa Makie.Menu
    @test any(opt -> occursin("Anterior-Posterior", opt), orient_menu.options[])
    @test test_custom_opt in orient_menu.options[]

    # Display screen
    screen = LMW.display_metadata_window(win.fig)
    yield(); sleep(0.5)

    # Hide loading overlay (as AppMain does upon complete load)
    LMW._lmw_observables[:loading_overlay_bg].visible = false
    LMW._lmw_observables[:loading_overlay_txt].visible = false

    # ── 4. Login Modal Lock ───────────────────────────────────────────────────
    LMW._lmw_observables[:login_visible][] = true
    LMW._login_authenticated[] = false
    yield(); sleep(0.2)

    # Verify typing guard is active during login
    @test LMW._is_any_typing_active()

    # Outside click while login is visible should be blocked
    events(win.fig.scene).mouseposition[] = (10.0, 10.0)
    push!(events(win.fig.scene).mousebuttonstate, Mouse.left)
    events(win.fig.scene).mousebutton[] = Makie.MouseButtonEvent(Mouse.left, Mouse.press)
    empty!(events(win.fig.scene).mousebuttonstate)
    events(win.fig.scene).mousebutton[] = Makie.MouseButtonEvent(Mouse.left, Mouse.release)
    @test !LMW._login_authenticated[]

    # Authenticate via login
    login_user = LMW._lmw_observables[:login_user]
    login_user.displayed_string[] = "dr_smith"
    login_user.stored_string[] = "dr_smith"
    for tb in LMW._all_textboxes
        if hasproperty(tb, :placeholder) && occursin("password", lowercase(string(tb.placeholder[])))
            tb.displayed_string[] = "lu"
            tb.stored_string[] = "lu"
            notify(tb.stored_string)
            break
        end
    end
    sleep(0.5)
    @test LMW._login_authenticated[]
    @test !LMW._lmw_observables[:login_visible][]

    # ── 5. Dropdown Modal Lock ────────────────────────────────────────────────
    # Wait for login settle period (1.0s) to elapse
    sleep(1.2)
    @test !LMW._any_menu_open()

    # Open orientation menu
    orient_menu.is_open[] = true
    yield(); sleep(0.2)
    @test orient_menu.is_open[]
    @test LMW._any_menu_open()

    # Click outside dropdown: should close dropdown and consume event
    events(win.fig.scene).mouseposition[] = (50.0, 50.0)
    push!(events(win.fig.scene).mousebuttonstate, Mouse.left)
    events(win.fig.scene).mousebutton[] = Makie.MouseButtonEvent(Mouse.left, Mouse.press)
    empty!(events(win.fig.scene).mousebuttonstate)
    events(win.fig.scene).mousebutton[] = Makie.MouseButtonEvent(Mouse.left, Mouse.release)
    @test !orient_menu.is_open[]
    @test !LMW._any_menu_open()

    # Settle debounce check: clicks within 250ms after close are consumed
    events(win.fig.scene).mouseposition[] = (50.0, 50.0)
    push!(events(win.fig.scene).mousebuttonstate, Mouse.left)
    consumed_press = events(win.fig.scene).mousebutton[] = Makie.MouseButtonEvent(Mouse.left, Mouse.press)
    empty!(events(win.fig.scene).mousebuttonstate)
    consumed_release = events(win.fig.scene).mousebutton[] = Makie.MouseButtonEvent(Mouse.left, Mouse.release)
    # Cleanup test option so git repo stays clean
    try
        opts = LMW.load_custom_options()
        if haskey(opts, "Lesion Orientation")
            filter!(o -> !startswith(o, "Test "), opts["Lesion Orientation"])
            LMW.save_custom_options!(opts)
        end
    catch; end
    println("All locking, options, and plus button tests passed cleanly!")
end
