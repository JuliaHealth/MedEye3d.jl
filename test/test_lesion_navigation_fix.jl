using Test
using GLMakie
using MedEye3d
import MedEye3d.LesionMetadataWindow as LMW
import MedEye3d.SegmentationDisplay.MakieEventHandlers as MEH

@testset "Up/Down Arrow Lesion Navigation & Dropdown Preservation" begin
    # Configure MEH state with 25 lesions at TP 0 (mimicking Pat3 + newly created Lesion 25)
    MEH.current_tp_index[] = 0
    MEH.volume_z_size[] = 326
    empty!(LMW._cached_lesion_ids)
    empty!(MEH.tp_segment_names)

    seg_names = Dict{Int, String}()
    for i in 1:24
        seg_names[i] = i == 5 ? "prostate" : (i == 24 ? "sartorius_right" : "femur_right")
        MEH.global_organ_mapping[][i] = seg_names[i]
    end
    seg_names[25] = "Thigh Adductor Compartment"
    MEH.global_organ_mapping[][25] = "Thigh Adductor Compartment"
    MEH.tp_segment_names[0] = seg_names

    initial_opts = String[]
    for i in 1:25
        if i == 1
            push!(initial_opts, "1: Femur Right")
        elseif i == 5
            # Deliberately generic name to trigger apply_state auto-rename
            push!(initial_opts, "5: Lesion 5")
        elseif i == 24
            push!(initial_opts, "24: Sartorius Right")
        elseif i == 25
            push!(initial_opts, "25: New Lesion")
        else
            push!(initial_opts, "$i: Femur Right ($i)")
        end
    end

    active_lesion = Observable(initial_opts[1])
    lesion_ids = Observable(copy(initial_opts))

    save_json_path = tempname() * ".json"
    win = LMW.create_metadata_window(active_lesion, lesion_ids, nothing; save_path = save_json_path)

    # Populate cache AFTER create_metadata_window (which clears _cached_lesion_ids)
    LMW._cached_lesion_ids[0] = collect(1:25)

    # Allow async DB init and GUI preload to complete
    t0 = time()
    while !get(LMW._lmw_observables, :gui_preload_done, Ref(false))[] && time() - t0 < 20.0
        sleep(0.1)
    end
    sleep(0.5)

    obs_next = LMW._lmw_observables[:obs_next_lesion]
    obs_prev = LMW._lmw_observables[:obs_prev_lesion]
    lesion_db = LMW._lmw_observables[:lesion_db]

    # ── Test 1: Updating lesion_db with only keys "1" and "25" must NOT strip 2..24 ──
    partial_db = Dict{String, Any}(
        "1" => Dict{String, Any}("_display_name" => "1: Femur Right", "BaseAnatomy" => "femur_right"),
        "25" => Dict{String, Any}("_display_name" => "25: Thigh Adductor Compartment", "BaseAnatomy" => "Thigh Adductor Compartment"),
    )
    lesion_db[] = partial_db
    @test length(lesion_ids[]) == 25
    parsed_all = [LMW.parse_lesion_id(o) for o in lesion_ids[]]
    @test parsed_all == collect(1:25)

    # ── Test 2: Stepping Up/Down traverses all 25 lesions sequentially without sticking ──
    # Start at Lesion 25 and press Up (obs_prev) all the way down through 24..1..25
    active_lesion[] = lesion_ids[][25]
    sleep(0.1)
    visited_prev = Int[]
    for step in 1:25
        obs_prev[] = obs_prev[] + 1
        push!(visited_prev, LMW.parse_lesion_id(active_lesion[]))
    end
    @test visited_prev == vcat(collect(24:-1:1), [25])

    # Verify auto-rename on generic Lesion 5 kept active_lesion[] synced with lesion_ids[][5]
    # (No stale "5: Lesion 5" mismatch causing reset to index 1 on next step)
    active_lesion[] = lesion_ids[][4]
    obs_next[] = obs_next[] + 1
    @test LMW.parse_lesion_id(active_lesion[]) == 5
    @test active_lesion[] in lesion_ids[]
    obs_next[] = obs_next[] + 1
    @test LMW.parse_lesion_id(active_lesion[]) == 6

    # Rapid forward traversal 1 -> 2 -> ... -> 25 -> 1 via obs_next
    active_lesion[] = lesion_ids[][1]
    visited_next = Int[]
    for step in 1:25
        obs_next[] = obs_next[] + 1
        push!(visited_next, LMW.parse_lesion_id(active_lesion[]))
    end
    @test visited_next == vcat(collect(2:25), [1])

    # ── Test 3: Self-healing if lesion_ids[] was ever truncated externally ──
    lesion_ids[] = ["1: Femur Right", "25: Thigh Adductor Compartment"]
    active_lesion[] = "25: Thigh Adductor Compartment"
    obs_prev[] = obs_prev[] + 1
    @test length(lesion_ids[]) == 25
    @test LMW.parse_lesion_id(active_lesion[]) == 24
end
