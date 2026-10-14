using Test
using Observables
using Dates
using JSON
using GLMakie

using MedEye3d
using MedEye3d.LesionMetadataWindow

@testset "Autosave, Edit History, and Widget Persistence Tests" begin
    # Create isolated temp files for testing
    tmp_json = tempname() * ".json"
    tmp_h5 = tempname() * ".h5"
    
    # ── Test 1: Edit History Auditing (Text/Value vs Image Segments) ──────────
    @testset "Audit Trail Separation" begin
        # Empty initial state
        st1 = Dict{String, Any}(
            "Certainty" => "10",
            "Alternative Hypothesis (False Positive)" => "None / Malignant Suspected",
            "BaseAnatomy" => "prostate gland"
        )
        
        # User modifies text/values
        st2 = copy(st1)
        st2["Certainty"] = "5"
        st2["Alternative Hypothesis (False Positive)"] = "Degenerative Osteoarthritis"
        st2["Comment"] = "Review needed"
        
        LesionMetadataWindow._append_edit_history!(st2, st1)
        @test haskey(st2, "_edit_history")
        hist = st2["_edit_history"]
        @test length(hist) == 1
        entry = hist[1]
        @test entry["action"] == "metadata_edit"
        @test !isempty(entry["user"])
        @test !isempty(entry["timestamp"])
        changes = entry["changes"]
        @test length(changes) == 3
        fields_changed = [c["field"] for c in changes]
        @test "Certainty" in fields_changed
        @test "Alternative Hypothesis (False Positive)" in fields_changed
        @test "Comment" in fields_changed
        
        # Now simulate an image segmentation mask edit
        LesionMetadataWindow._append_seg_edit_to_history!(st2, "jm", "paint_brush")
        @test st2["_last_seg_edit_by"] == "jm"
        @test st2["_last_seg_edit_tool"] == "paint_brush"
        @test !isempty(st2["_last_seg_edit_at"])
        # Crucial check: _last_seg_edit_* fields must NOT be added to _edit_history!
        @test length(st2["_edit_history"]) == 1
    end

    # ── Test 2: Searchable Menu Selection Preservation ───────────────────────
    @testset "Searchable Menu Trigeminal Ganglion & Schema Options" begin
        # Verify large ontology menu preserves selected item even if not in top 50
        opts = ["Organ_$i" for i in 1:1000]
        push!(opts, "Trigeminal Ganglion")
        
        fig = Figure()
        m = LesionMetadataWindow.searchable_menu(fig[1, 1], 1, 1; options = opts, default = 1)
        @test length(m.options[]) == 51 # 50 + "(type to filter...)"
        
        # Programmatically select "Trigeminal Ganglion"
        LesionMetadataWindow._set_menu_selection_string!(m, "Trigeminal Ganglion")
        @test m.selection[] == "Trigeminal Ganglion"
        
        # Simulate filter query reset (user opened and closed menu or cleared filter)
        LesionMetadataWindow._apply_searchable_filter!(m, "", opts)
        # Verify selection is NOT wiped out!
        @test m.selection[] == "Trigeminal Ganglion"
        @test "Trigeminal Ganglion" in m.options[]
        
        # Verify schema menu (small option set <= 150 items) is never truncated
        schema_opts = ["Opt_$i" for i in 1:30]
        m_schema = LesionMetadataWindow.searchable_menu(fig[2, 1], 1, 1; options = schema_opts, default = 1)
        @test length(m_schema.options[]) == 30 # Not truncated to 25!
        LesionMetadataWindow._set_menu_selection_string!(m_schema, "Opt_28")
        @test m_schema.selection[] == "Opt_28"
        LesionMetadataWindow._apply_searchable_filter!(m_schema, "", schema_opts)
        @test m_schema.selection[] == "Opt_28"
    end

    # ── Test 3: Full Metadata Window Persistence & Restart Simulation ────────
    @testset "Window Creation, Autosave, and Restart Retrieval" begin
        init_db = Dict{String, Any}(
            "1" => Dict{String, Any}(
                "_display_name" => "1: Right Sartorius Muscle",
                "BaseAnatomy" => "Trigeminal Ganglion",
                "BaseAnatomySide" => "Right",
                "_user_edited_anatomy" => true,
                "LesionType" => "Technical Artifact",
                "Certainty" => "5",
                "Alternative Hypothesis (False Positive)" => "Degenerative Osteoarthritis",
                "Inner Texture / Density / Attenuation" => "Sclerotic / Blastic / Ivory (>1000 HU)",
                "Border and Margin" => "Reactive Sclerotic Rim (Sharp Outer, Fuzzy Inner)"
            )
        )
        LesionMetadataWindow.save_annotations(init_db, tmp_json)
        
        active_les = Observable("1: Right Sartorius Muscle")
        les_ids = Observable(["1: Right Sartorius Muscle"])
        
        # Session 1: Create window
        win = LesionMetadataWindow.create_metadata_window(active_les, les_ids, nothing; save_path = tmp_json)
        sleep(0.5)
        
        db1 = win.lesion_db[]
        @test haskey(db1, "1")
        @test db1["1"]["Certainty"] == "5"
        @test db1["1"]["Alternative Hypothesis (False Positive)"] == "Degenerative Osteoarthritis"
        @test db1["1"]["BaseAnatomy"] == "Trigeminal Ganglion"
        
        # Check field widgets
        fw = LesionMetadataWindow._lmw_observables
        cert_sl = fw[:field_widgets]["Certainty"]
        alt_hyp_menu = fw[:field_widgets]["Alternative Hypothesis (False Positive)"]
        base_anat_menu = fw[:menu_base_anat]
        
        @test cert_sl.value[] == 5
        @test alt_hyp_menu.selection[] == "Degenerative Osteoarthritis"
        @test base_anat_menu.selection[] == "Trigeminal Ganglion"
        
        # User changes Certainty to 8 and Alternative Hypothesis to "Brown Adipose Tissue Uptake"
        cert_sl.selected_index[] = 9  # In 0:10 range, index 9 = value 8
        set_close_to!(cert_sl, 8)
        @test cert_sl.value[] == 8
        
        LesionMetadataWindow._set_menu_selection_string!(alt_hyp_menu, "Brown Adipose Tissue Uptake")
        @test alt_hyp_menu.selection[] == "Brown Adipose Tissue Uptake"
        
        # Trigger autosave and wait for debounced task
        LesionMetadataWindow.request_autosave()
        sleep(1.0) # Debounce is 300ms + async save
        
        # Wait a moment for background save to finish
        sleep(1.0)
        
        # Flush lesion to disk explicitly as atexit does
        LesionMetadataWindow.flush_current_lesion!(1)
        sleep(0.5)
        
        # Read saved json from disk
        saved_on_disk = LesionMetadataWindow.load_annotations(tmp_json)
        @test haskey(saved_on_disk, "1")
        @test saved_on_disk["1"]["Certainty"] == "8"
        @test saved_on_disk["1"]["Alternative Hypothesis (False Positive)"] == "Brown Adipose Tissue Uptake"
        @test saved_on_disk["1"]["BaseAnatomy"] == "Trigeminal Ganglion"
        @test haskey(saved_on_disk["1"], "_edit_history")
        
        # Session 2 (RESTART SIMULATION): Create a brand new window from the saved file
        active_les2 = Observable("1: Right Sartorius Muscle")
        les_ids2 = Observable(["1: Right Sartorius Muscle"])
        win2 = LesionMetadataWindow.create_metadata_window(active_les2, les_ids2, nothing; save_path = tmp_json)
        sleep(0.5)
        
        fw2 = LesionMetadataWindow._lmw_observables
        cert_sl2 = fw2[:field_widgets]["Certainty"]
        alt_hyp_menu2 = fw2[:field_widgets]["Alternative Hypothesis (False Positive)"]
        base_anat_menu2 = fw2[:menu_base_anat]
        
        # Verify all values were properly retrieved and displayed in UI widgets!
        @test cert_sl2.value[] == 8
        @test alt_hyp_menu2.selection[] == "Brown Adipose Tissue Uptake"
        @test base_anat_menu2.selection[] == "Trigeminal Ganglion"
    end
    
    # Cleanup temp files
    rm(tmp_json, force=true)
    rm(tmp_h5, force=true)
end
