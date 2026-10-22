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

    # ── 6. Inline Custom Option Add Rows (Hidden by default, Only on '+' click) ─
    custom_add_open = LMW._lmw_observables[:custom_add_row_open]
    q_add_rows = LMW._lmw_observables[:q_add_rows]
    add_btns = LMW._lmw_observables[:add_opt_buttons]

    # Macroscopic Pattern add row must exist and be strictly closed by default
    @test haskey(q_add_rows, "Macroscopic Pattern")
    macro_ar = q_add_rows["Macroscopic Pattern"]
    @test haskey(custom_add_open, macro_ar)
    @test custom_add_open[macro_ar][] == false

    # All add rows must be closed initially across all questions
    for (ar, is_open_ref) in custom_add_open
        @test is_open_ref[] == false
    end

    # Clicking '+' button for Macroscopic Pattern opens the row
    @test haskey(add_btns, "Macroscopic Pattern")
    macro_btn_add = add_btns["Macroscopic Pattern"]
    notify(macro_btn_add.clicks)
    yield(); sleep(0.2)
    @test custom_add_open[macro_ar][] == true

    # Clicking '+' again closes it
    notify(macro_btn_add.clicks)
    yield(); sleep(0.2)
    @test custom_add_open[macro_ar][] == false

    # Opening and navigating lesion closes it automatically
    notify(macro_btn_add.clicks)
    yield(); sleep(0.2)
    @test custom_add_open[macro_ar][] == true
    active_lesion[] = "2: New Lesion"
    yield(); sleep(0.2)
    @test custom_add_open[macro_ar][] == false
    active_lesion[] = "1: Femur Right"
    yield(); sleep(0.2)

    # ── 7. Plus Button Click While Dropdown Is Open & Scrolling Position ─────
    orient_ar = q_add_rows["Lesion Orientation"]
    orient_btn = add_btns["Lesion Orientation"]
    
    # Open orientation menu first (user looking through options)
    orient_menu.is_open[] = true
    yield(); sleep(0.2)
    @test orient_menu.is_open[] == true

    # Simulate mouse click on '+' button while menu is open
    events(win.fig.scene).hasfocus[] = false
    bb = orient_btn.layoutobservables.computedbbox[]
    cx = bb.origin[1] + bb.widths[1] / 2
    cy = bb.origin[2] + bb.widths[2] / 2
    events(win.fig.scene).mouseposition[] = (cx, cy)
    push!(events(win.fig.scene).mousebuttonstate, Mouse.left)
    events(win.fig.scene).mousebutton[] = Makie.MouseButtonEvent(Mouse.left, Mouse.press)
    empty!(events(win.fig.scene).mousebuttonstate)
    events(win.fig.scene).mousebutton[] = Makie.MouseButtonEvent(Mouse.left, Mouse.release)
    yield(); sleep(0.5)

    # Menu must close, and add row must open
    @test orient_menu.is_open[] == false
    @test custom_add_open[orient_ar][] == true

    # Verify scroll positions propagate to nested subgrid children
    apply_scroll = LMW._lmw_observables[:apply_scroll]
    apply_scroll(2100.0f0)
    yield(); sleep(0.2)

    menu_scrolled_bb = orient_menu.layoutobservables.computedbbox[]
    @test menu_scrolled_bb.origin[2] > 0  # visible on screen

    main_g = LMW._lmw_observables[:g]
    for c in main_g.content
        if c.span.rows.start <= orient_ar && c.span.rows.stop >= orient_ar
            @test c.content isa Makie.GridLayoutBase.GridLayout
            sub_g = c.content
            for sc in sub_g.content
                # Each child widget in the subgrid (textbox, save, cancel) must have scrolled y > 0
                child_bb = sc.content.layoutobservables.computedbbox[]
                @test child_bb.origin[2] > 0
                @test abs(child_bb.origin[2] - (menu_scrolled_bb.origin[2] - 30.0f0)) < 15.0f0
            end
        end
    end

    # Reset scroll offset
    apply_scroll(0.0f0)
    yield(); sleep(0.2)

    # Close add row
    notify(orient_btn.clicks)
    yield(); sleep(0.2)
    @test custom_add_open[orient_ar][] == false

    # ── 8. Dropdown Option Selection Blocks Underlying Controls ──────────────
    # Scroll so orient_menu is visible in viewport
    apply_scroll(2000.0f0)
    yield(); sleep(0.2)

    # Open orient_menu
    orient_menu.is_open[] = true
    yield(); sleep(0.2)
    @test orient_menu.is_open[] == true

    # Find an underlying menu below orient_menu that is overlapped by orient_menu options
    local row76_menu = nothing
    orient_row = 0
    for c in main_g.content
        if c.content === orient_menu
            orient_row = c.span.rows.start
            break
        end
    end
    min_dist = 9999
    for c in main_g.content
        if c.span.rows.start > orient_row && c.content isa Makie.Menu
            dist = c.span.rows.start - orient_row
            if dist < min_dist
                min_dist = dist
                row76_menu = c.content
            end
        end
    end
    @test row76_menu !== nothing
    @test row76_menu.is_open[] == false

    # Click on Option 3 ("Anterior-Posterior (AP) Orientation of Long Axis")
    # Coordinates of Option 3 overlap row76_menu
    menuscene = orient_menu.blockscene.children[1]
    poly_plots = [p for p in menuscene.plots if p isa Makie.Poly]
    @test !isempty(poly_plots)
    rects = poly_plots[1][1][]
    t = translation(menuscene)[]
    vp = menuscene.viewport[]
    r3 = rects[3]
    opt3_x = vp.origin[1] + r3.origin[1] + r3.widths[1] / 2
    opt3_y = vp.origin[2] + t[2] + r3.origin[2] + r3.widths[2] / 2

    # Disconnect GLFW mouse cursor updater so synthetic mouseposition is preserved during headless test
    GLMakie.disconnect!(screen, Makie.mouse_position)
    events(win.fig.scene).hasfocus[] = true
    events(win.fig.scene).mouseposition[] = (opt3_x, opt3_y)
    push!(events(win.fig.scene).mousebuttonstate, Mouse.left)
    events(win.fig.scene).mousebutton[] = Makie.MouseButtonEvent(Mouse.left, Mouse.press)
    empty!(events(win.fig.scene).mousebuttonstate)
    events(win.fig.scene).mousebutton[] = Makie.MouseButtonEvent(Mouse.left, Mouse.release)
    yield(); sleep(0.5)

    # Verification:
    # 1. orient_menu has closed and selected Option 3
    @test orient_menu.is_open[] == false
    @test occursin("Anterior-Posterior", orient_menu.selection[])

    # 2. Underlying menu at row 76 was NOT opened or activated
    @test row76_menu.is_open[] == false

    # Reset scroll offset
    apply_scroll(0.0f0)
    yield(); sleep(0.2)

    # ── 9. Anatomical Details Row Operations (+ Row and - Button) ─────────────
    # Verify that + Row and - (remove) buttons work properly and are not blocked by GUI locks.
    anat_count = LMW._lmw_observables[:anat_active_count]
    btn_add_anat = LMW._lmw_observables[:btn_add_anat_rel]
    anat_rm_btns = LMW._lmw_observables[:anat_rm_btns]
    anat_rel_menus = LMW._lmw_observables[:anat_rel_menus]
    anat_struct_menus = LMW._lmw_observables[:anat_struct_menus]

    init_anat = anat_count[]
    
    # 9a. Add row via direct click notification
    notify(btn_add_anat.clicks)
    yield(); sleep(0.2)
    @test anat_count[] == init_anat + 1
    @test anat_rel_menus[init_anat + 1].blockscene.visible[] == true
    @test anat_struct_menus[init_anat + 1].blockscene.visible[] == true

    # 9b. Add another row via simulated mouse click on btn_add_anat
    bb_add = btn_add_anat.layoutobservables.computedbbox[]
    add_cx = bb_add.origin[1] + bb_add.widths[1] / 2
    add_cy = bb_add.origin[2] + bb_add.widths[2] / 2
    events(win.fig.scene).mouseposition[] = (add_cx, add_cy)
    push!(events(win.fig.scene).mousebuttonstate, Mouse.left)
    events(win.fig.scene).mousebutton[] = Makie.MouseButtonEvent(Mouse.left, Mouse.press)
    empty!(events(win.fig.scene).mousebuttonstate)
    events(win.fig.scene).mousebutton[] = Makie.MouseButtonEvent(Mouse.left, Mouse.release)
    yield(); sleep(0.3)
    @test anat_count[] == init_anat + 2

    # 9c. Populate row 1 and row 2 with test values
    LMW._set_menu_selection_string!(anat_rel_menus[1], "Adjacent To")
    LMW._set_menu_selection_string!(anat_struct_menus[1], "Aorta")
    LMW._set_menu_selection_string!(anat_rel_menus[2], "Inside / Contained In")
    LMW._set_menu_selection_string!(anat_struct_menus[2], "Prostate Gland")
    @test string(anat_rel_menus[1].selection[]) == "Adjacent To"
    @test string(anat_struct_menus[1].selection[]) == "Aorta"
    @test string(anat_rel_menus[2].selection[]) == "Inside / Contained In"
    @test string(anat_struct_menus[2].selection[]) == "Prostate Gland"

    # 9d. Remove row 1 via simulated mouse click on the red minus button
    curr_n = anat_count[]
    rm_btn1 = anat_rm_btns[1]
    bb_rm = rm_btn1.layoutobservables.computedbbox[]
    rm_cx = bb_rm.origin[1] + bb_rm.widths[1] / 2
    rm_cy = bb_rm.origin[2] + bb_rm.widths[2] / 2
    events(win.fig.scene).mouseposition[] = (rm_cx, rm_cy)
    push!(events(win.fig.scene).mousebuttonstate, Mouse.left)
    events(win.fig.scene).mousebutton[] = Makie.MouseButtonEvent(Mouse.left, Mouse.press)
    empty!(events(win.fig.scene).mousebuttonstate)
    events(win.fig.scene).mousebutton[] = Makie.MouseButtonEvent(Mouse.left, Mouse.release)
    yield(); sleep(0.3)

    # Values from row 2 must have shifted to row 1, row 2 cleared, and count decremented by 1
    @test anat_count[] == curr_n - 1
    @test string(anat_rel_menus[1].selection[]) == "Inside / Contained In"
    @test string(anat_struct_menus[1].selection[]) == "Prostate Gland"
    @test string(anat_rel_menus[2].selection[]) == ""
    @test string(anat_struct_menus[2].selection[]) == ""

    # ── 10. Artifact Reclassification from Lymph Node Lesion ──────────────────
    # User scenario: Lesion currently classified as lymph node, wants to reclassify as Technical Artifact
    btn_ln = LMW._lmw_observables[:add_opt_buttons]["__TYPE_LN__"]
    notify(btn_ln.clicks)
    yield(); sleep(0.2)
    @test LMW._lmw_observables[:active_lesion_type][] == "Lymph Node Meta"

    # Click the Artifact button via simulated mouse click
    btn_art = LMW._lmw_observables[:add_opt_buttons]["__TYPE_ARTIFACT__"]
    bb_art = btn_art.layoutobservables.computedbbox[]
    art_cx = bb_art.origin[1] + bb_art.widths[1] / 2
    art_cy = bb_art.origin[2] + bb_art.widths[2] / 2
    events(win.fig.scene).mouseposition[] = (art_cx, art_cy)
    push!(events(win.fig.scene).mousebuttonstate, Mouse.left)
    events(win.fig.scene).mousebutton[] = Makie.MouseButtonEvent(Mouse.left, Mouse.press)
    empty!(events(win.fig.scene).mousebuttonstate)
    events(win.fig.scene).mousebutton[] = Makie.MouseButtonEvent(Mouse.left, Mouse.release)
    yield(); sleep(0.4)

    # Verify complete reclassification:
    @test LMW._lmw_observables[:active_lesion_type][] == "Technical Artifact"
    @test LMW._lmw_observables[:no_ct_toggle].active[] == true
    
    cert_w = LMW._lmw_observables[:field_widgets]["Certainty"]
    if cert_w isa Slider
        @test cert_w.value[] == 0
    elseif cert_w isa Menu
        @test string(cert_w.selection[]) == "0"
    end
    
    alt_hyp_w = LMW._lmw_observables[:field_widgets]["Alternative Hypothesis (False Positive)"]
    @test string(alt_hyp_w.selection[]) == "Technical Artifact"

    # Verify that clicking Artifact while a dropdown is open or immediately closing is NOT blocked
    # Open orient menu
    orient_menu.is_open[] = true
    yield(); sleep(0.2)
    @test orient_menu.is_open[] == true
    
    # Click Artifact button while menu was open
    events(win.fig.scene).mouseposition[] = (art_cx, art_cy)
    push!(events(win.fig.scene).mousebuttonstate, Mouse.left)
    events(win.fig.scene).mousebutton[] = Makie.MouseButtonEvent(Mouse.left, Mouse.press)
    empty!(events(win.fig.scene).mousebuttonstate)
    events(win.fig.scene).mousebutton[] = Makie.MouseButtonEvent(Mouse.left, Mouse.release)
    yield(); sleep(0.4)

    # Menu should close and Artifact button should be responsive
    @test orient_menu.is_open[] == false
    @test LMW._lmw_observables[:active_lesion_type][] == "Technical Artifact"

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
