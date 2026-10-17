using Test
using GLMakie
using Observables
using JSON
using HDF5

using MedEye3d
import MedEye3d.LesionMetadataWindow as LMW
import MedEye3d.SegmentationDisplay.MakieEventHandlers as MEH

@testset "Anatomy Persistence and UI Overlap Suite" begin

    function count_visible_overlaps(g)
        vis_widgets = []
        for (i, c) in enumerate(g.content)
            w = c.content
            vis = true
            if hasproperty(w, :visible) && !w.visible[]
                vis = false
            elseif hasproperty(w, :blockscene) && !w.blockscene.visible[]
                vis = false
            end
            if vis
                bb = w.layoutobservables.computedbbox[]
                if bb.widths[1] > 1 && bb.widths[2] > 1
                    push!(vis_widgets, (i=i, w=w, rows=c.span.rows, cols=c.span.cols, bbox=bb))
                end
            end
        end

        n_overlaps = 0
        for i in 1:length(vis_widgets)
            w1 = vis_widgets[i]
            bb1 = w1.bbox
            for j in (i+1):length(vis_widgets)
                w2 = vis_widgets[j]
                bb2 = w2.bbox
                
                x_overlap = max(0.0f0, min(bb1.origin[1] + bb1.widths[1], bb2.origin[1] + bb2.widths[1]) - max(bb1.origin[1], bb2.origin[1]))
                y_overlap = max(0.0f0, min(bb1.origin[2] + bb1.widths[2], bb2.origin[2] + bb2.widths[2]) - max(bb1.origin[2], bb2.origin[2]))
                
                if x_overlap > 2.0f0 && y_overlap > 2.0f0
                    n_overlaps += 1
                end
            end
        end
        return n_overlaps
    end

    @testset "1. Zero Overlaps Across Dynamic States" begin
        MEH.current_tp_index[] = 0
        MEH.volume_z_size[] = 326
        MEH.global_organ_mapping[][1] = "femur_right"
        MEH.tp_segment_names[0] = Dict(1 => "Femur Right")

        active_lesion = Observable("1: Femur Right")
        lesion_ids = Observable(["1: Femur Right"])

        win = LMW.create_metadata_window(active_lesion, lesion_ids, nothing)
        sleep(1.2) # wait for initial preload

        main_layout = win.fig.layout.content[1].content
        g = nothing
        for c in main_layout.content
            if c.span.rows.start == 2 && c.span.cols.start == 1
                g = c.content
                break
            end
        end
        @test g !== nothing

        # Base state: 0 anatomy detail rows
        @test count_visible_overlaps(g) == 0

        # Dynamic test: 1, 2, 3, 4, 5, 6 anatomy detail rows
        count_obs = LMW._lmw_observables[:anat_active_count]
        for n in [1, 2, 3, 4, 5, 6, 0]
            count_obs[] = n
            sleep(0.1)
            @test count_visible_overlaps(g) == 0
        end

        # Dynamic test: Switch lesion types (Prostate, Bone Meta, Organ Meta, Lymph Node, Artifact)
        update_fn = LMW._lmw_observables[:update_type_buttons]
        for t in ["Prostate", "Bone Meta", "Organ Meta", "Lymph Node Meta", "Technical Artifact"]
            update_fn(t)
            sleep(0.1)
            @test count_visible_overlaps(g) == 0
        end

        # Dynamic test: Toggle No CT Correlate
        no_ct_toggle = LMW._lmw_observables[:no_ct_toggle]
        no_ct_toggle.active[] = true
        sleep(0.1)
        @test count_visible_overlaps(g) == 0

        no_ct_toggle.active[] = false
        sleep(0.1)
        @test count_visible_overlaps(g) == 0
    end

    @testset "2. HDF5 Persistence on Refresh Anatomy Button" begin
        tmp_h5 = tempname() * ".h5"
        # Create initial test HDF5 with volumes and _meta_
        HDF5.h5open(tmp_h5, "w") do f
            m = create_group(f, "_meta_")
            m["organ_mapping"] = JSON.json(Dict("0" => Dict("1" => "femur_right")), 2)
            m["segment_names.json"] = JSON.json(Dict("0" => Dict("1" => "1: Femur Right")), 2)
            # Add a dummy study group to simulate active dataset
            b = create_group(f, "BASELINE")
            b["ct"] = zeros(Float32, 4, 4, 4)
        end

        MEH.h5_path_ref[] = tmp_h5
        MEH.current_tp_index[] = 0
        MEH.global_organ_mapping[][1] = "femur_right"
        MEH.tp_segment_names[0] = Dict(1 => "1: Femur Right")
        atlas = fill(Int16(10), 10, 10, 10)
        MEH.global_ts_atlas[] = atlas
        MEH.global_ts_names[] = Dict(10 => "femur_right")
        MEH.lesion_centroids_cache[(0, 1)] = [5, 5, 5]
        MEH.lesion_centroids_cache[1] = [5, 5, 5]

        active_lesion = Observable("1: Femur Right")
        lesion_ids = Observable(["1: Femur Right"])

        win = LMW.create_metadata_window(active_lesion, lesion_ids, nothing)
        sleep(1.2)

        btn_refresh = LMW._lmw_observables[:btn_refresh_anat]
        @test btn_refresh !== nothing

        # Simulate user clicking Refresh Anatomy
        notify(btn_refresh.clicks)
        sleep(0.5)

        # Verify in-memory state updated
        @test haskey(MEH.tp_organ_mapping, 0)
        @test haskey(MEH.tp_organ_mapping[0], 1)
        @test haskey(MEH.tp_segment_names[0], 1)

        # Verify disk HDF5 state updated
        HDF5.h5open(tmp_h5, "r") do f
            @test haskey(f, "_meta_")
            @test haskey(f["_meta_"], "organ_mapping")
            @test haskey(f["_meta_"], "segment_names.json")
            @test haskey(f["_meta_"], "lesion_annotations.json")

            om = JSON.parse(read(f["_meta_/organ_mapping"]))
            @test haskey(om, "0")
            @test haskey(om["0"], "1")
            @test !isempty(om["0"]["1"])

            sn = JSON.parse(read(f["_meta_/segment_names.json"]))
            @test haskey(sn, "0")
            @test haskey(sn["0"], "1")
            @test !isempty(sn["0"]["1"])

            la = JSON.parse(read(f["_meta_/lesion_annotations.json"]))
            @test haskey(la, "1")
            @test haskey(la["1"], "BaseAnatomy")
            @test !isempty(la["1"]["BaseAnatomy"])
        end

        rm(tmp_h5; force=true)
    end

    @testset "3. HDF5 Persistence on Manual Menu Edits" begin
        tmp_h5 = tempname() * ".h5"
        HDF5.h5open(tmp_h5, "w") do f
            m = create_group(f, "_meta_")
            m["organ_mapping"] = JSON.json(Dict("0" => Dict("1" => "unknown")), 2)
            m["segment_names.json"] = JSON.json(Dict("0" => Dict("1" => "1: Unknown")), 2)
            b = create_group(f, "BASELINE")
            b["ct"] = zeros(Float32, 4, 4, 4)
        end

        MEH.h5_path_ref[] = tmp_h5
        MEH.current_tp_index[] = 0
        MEH.global_organ_mapping[][1] = "unknown"
        MEH.tp_segment_names[0] = Dict(1 => "1: Unknown")

        active_lesion = Observable("1: Unknown")
        lesion_ids = Observable(["1: Unknown"])

        win = LMW.create_metadata_window(active_lesion, lesion_ids, nothing)
        sleep(1.2)

        menu_ba = LMW._lmw_observables[:menu_base_anat]
        menu_sd = LMW._lmw_observables[:menu_side]
        @test menu_ba !== nothing
        @test menu_sd !== nothing

        # User manually selects "Liver" and "Right"
        menu_ba.selection[] = "Liver"
        menu_sd.selection[] = "Right"

        # Wait for debounced autosave
        sleep(0.8)

        # Check in-memory caches
        @test MEH.tp_organ_mapping[0][1] == "Liver"
        @test MEH.global_organ_mapping[][1] == "Liver"
        @test occursin("Liver", MEH.tp_segment_names[0][1])
        @test occursin("Right", MEH.tp_segment_names[0][1])

        # Check dropdown label
        @test occursin("Right Liver", active_lesion[])
        @test any(opt -> occursin("Right Liver", opt), lesion_ids[])

        # Direct verification of HDF5 file on disk
        HDF5.h5open(tmp_h5, "r") do f
            om = JSON.parse(read(f["_meta_/organ_mapping"]))
            @test om["0"]["1"] == "Liver"

            sn = JSON.parse(read(f["_meta_/segment_names.json"]))
            @test occursin("Liver", sn["0"]["1"])
            @test occursin("Right", sn["0"]["1"])

            la = JSON.parse(read(f["_meta_/lesion_annotations.json"]))
            @test la["1"]["BaseAnatomy"] == "Liver"
            @test la["1"]["BaseAnatomySide"] == "Right"
        end

        # Test loading back through load_annotations_hdf5
        loaded = LMW.load_annotations_hdf5(tmp_h5)
        @test haskey(loaded, "1")
        @test loaded["1"]["BaseAnatomy"] == "Liver"
        @test loaded["1"]["BaseAnatomySide"] == "Right"

        rm(tmp_h5; force=true)
    end

    @testset "4. Active Dataset Protection" begin
        # Guard test: ensure save_annotations_hdf5 does NOT truncate an active dataset with volumes!
        tmp_h5 = tempname() * ".h5"
        HDF5.h5open(tmp_h5, "w") do f
            create_group(f, "_meta_")
            b = create_group(f, "BASELINE")
            b["ct_volume"] = fill(42.0f0, 10, 10, 10)
        end

        MEH.h5_path_ref[] = tmp_h5
        test_db = Dict("1" => Dict("BaseAnatomy" => "Spleen", "LesionType" => "Organ Meta"))
        
        # Call save_annotations_hdf5 with active dataset path
        LMW.save_annotations_hdf5(test_db, tmp_h5)

        # Verify volumes still exist (not truncated by "w" mode!)
        HDF5.h5open(tmp_h5, "r") do f
            @test haskey(f, "BASELINE")
            @test haskey(f["BASELINE"], "ct_volume")
            @test sum(read(f["BASELINE/ct_volume"])) == 42.0f0 * 1000
            @test haskey(f, "_meta_/lesion_annotations.json")
        end

        rm(tmp_h5; force=true)
        MEH.h5_path_ref[] = ""
    end

    @testset "5. Trigeminal Ganglion Manual Edit Across App Restart" begin
        tmp_json = tempname() * ".json"
        tmp_h5 = replace(tmp_json, ".json" => ".h5")

        HDF5.h5open(tmp_h5, "w") do f
            m = create_group(f, "_meta_")
            m["segment_names.json"] = JSON.json(Dict("0" => Dict("1" => "Segment 1")), 2)
            b = create_group(f, "BASELINE")
            b["ct"] = zeros(Float32, 4, 4, 4)
            b["mask"] = zeros(Int16, 4, 4, 4)
            b["mask"][2, 2, 2] = 1
        end

        MEH.h5_path_ref[] = tmp_h5
        MEH.current_tp_index[] = 0
        MEH.volume_z_size[] = 326
        MEH.tp_segment_names[0] = Dict(1 => "Segment 1")
        empty!(MEH.tp_organ_mapping)
        empty!(MEH.global_organ_mapping[])
        MEH.global_organ_mapping[][1] = ""

        active_lesion = Observable("1: Segment 1")
        lesion_ids = Observable(["1: Segment 1"])

        win = LMW.create_metadata_window(active_lesion, lesion_ids, nothing; save_path = tmp_json)
        sleep(1.2)

        menu_ba = LMW._lmw_observables[:menu_base_anat]
        @test menu_ba !== nothing

        # User changes anatomy to Trigeminal Ganglion
        menu_ba.selection[] = "Trigeminal Ganglion"
        sleep(1.0) # wait for autosave

        @test active_lesion[] == "1: Trigeminal Ganglion"
        @test any(opt -> opt == "1: Trigeminal Ganglion", lesion_ids[])
        @test menu_ba.selection[] == "Trigeminal Ganglion"

        # Verify disk HDF5 state
        HDF5.h5open(tmp_h5, "r") do f
            la = JSON.parse(read(f["_meta_/lesion_annotations.json"]))
            @test la["1"]["BaseAnatomy"] == "Trigeminal Ganglion"
            @test la["1"]["_display_name"] == "1: Trigeminal Ganglion"
            sn = JSON.parse(read(f["_meta_/segment_names.json"]))
            @test sn["0"]["1"] == "Trigeminal Ganglion" # No duplicate '1: ' prefix stored
        end

        # AppMain reload simulation
        persisted_annotations = Dict{String, Any}()
        HDF5.h5open(tmp_h5, "r") do h5_init
            if haskey(h5_init, "_meta_") && haskey(h5_init, "_meta_/lesion_annotations.json")
                persisted_annotations = JSON.parse(read(h5_init["_meta_/lesion_annotations.json"]))
            end
            if haskey(h5_init, "_meta_") && haskey(h5_init, "_meta_/segment_names.json")
                raw_sn = JSON.parse(read(h5_init["_meta_/segment_names.json"]))
                for (tp_str, sdict) in raw_sn
                    tp_i = parse(Int, tp_str)
                    target = get!(MEH.tp_segment_names, tp_i, Dict{Int, String}())
                    for (sid_str, sname) in sdict
                        target[parse(Int, sid_str)] = string(sname)
                    end
                end
            end
        end

        lesion_ids_ints = [1]
        tp0_sn = get(MEH.tp_segment_names, 0, Dict{Int, String}())
        lesion_list = map(lesion_ids_ints) do i
            seg_int = Int(i)
            display_name = ""
            if haskey(persisted_annotations, string(seg_int))
                p_entry = persisted_annotations[string(seg_int)]
                if p_entry isa AbstractDict
                    p_disp = get(p_entry, "_display_name", "")
                    p_anat = get(p_entry, "BaseAnatomy", "")
                    p_side = get(p_entry, "BaseAnatomySide", "")
                    if !isempty(p_disp) && !startswith(p_disp, "Lesion ")
                        clean_disp = replace(p_disp, Regex("^$(seg_int):\\s*") => "")
                        display_name = clean_disp
                    elseif !isempty(p_anat)
                        clean_a = replace(p_anat, r"\s*\(UBERON\)" => "")
                        clean_a = titlecase(replace(clean_a, "_" => " "))
                        display_name = (isempty(p_side) || p_side == "None" ? "" : "$p_side ") * clean_a
                    end
                end
            end
            if isempty(display_name) && haskey(tp0_sn, seg_int) && !isempty(tp0_sn[seg_int])
                sn = replace(tp0_sn[seg_int], Regex("^$(seg_int):\\s*") => "")
                if !startswith(sn, "unknown_") && !startswith(sn, "Segment") && !startswith(sn, "artificial_")
                    display_name = sn
                end
            end
            display_name = isempty(display_name) ? "Lesion $seg_int" : display_name
            return "$seg_int: $display_name"
        end

        @test lesion_list == ["1: Trigeminal Ganglion"]

        # Session 2 GUI launch with reloaded lesion_list
        active_lesion2 = Observable(lesion_list[1])
        lesion_ids2 = Observable(lesion_list)

        win2 = LMW.create_metadata_window(active_lesion2, lesion_ids2, nothing; save_path = tmp_json)
        sleep(1.5)

        menu_ba2 = LMW._lmw_observables[:menu_base_anat]
        @test active_lesion2[] == "1: Trigeminal Ganglion"
        @test lesion_ids2[] == ["1: Trigeminal Ganglion"]
        @test menu_ba2.selection[] == "Trigeminal Ganglion"

        # Verify Macro and Other Structural labels fit within column 1 with no horizontal overflow
        g2 = LMW._lmw_observables[:g]
        g2_origin_x = g2.layoutobservables.computedbbox[].origin[1]
        for c in g2.content
            w = c.content
            if w isa Label
                t = string(w.text[])
                if occursin("Macro", t) || occursin("Other Structural", t)
                    bb = w.layoutobservables.computedbbox[]
                    @test bb.origin[1] >= g2_origin_x # Within column 1
                    @test (bb.origin[1] - g2_origin_x) + bb.widths[1] <= 118.0f0 # Contained within column 1 (Fixed 115)
                end
            end
        end

        rm(tmp_json; force=true)
        rm(tmp_h5; force=true)
        MEH.h5_path_ref[] = ""
    end

end
