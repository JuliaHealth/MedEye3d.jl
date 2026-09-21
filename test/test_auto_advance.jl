using Test
using MedEye3d
using MedEye3d.MakieEvents
using Observables

const MEH = MedEye3d.SegmentationDisplay.MakieEventHandlers
const LMW = MedEye3d.LesionMetadataWindow

@testset "Auto-Advance and TP Navigation Tests" begin

    @testset "1. TP Labels and Index State" begin
        # MEH exposes current_tp_index and tp_labels
        @test MEH.current_tp_index isa Ref{Int}
        @test MEH.tp_labels isa Dict{Int, String}

        # Can set and read current TP index
        MEH.current_tp_index[] = 0
        @test MEH.current_tp_index[] == 0
        MEH.current_tp_index[] = 2
        @test MEH.current_tp_index[] == 2
        MEH.current_tp_index[] = 0  # Reset
    end

    @testset "2. TP Label Registration" begin
        # Can add and read TP labels
        empty!(MEH.tp_labels)
        MEH.tp_labels[0] = "PET TP0"
        MEH.tp_labels[1] = "PET TP1"
        MEH.tp_labels[2] = "PET TP2"

        @test MEH.tp_labels[0] == "PET TP0"
        @test MEH.tp_labels[1] == "PET TP1"
        @test MEH.tp_labels[2] == "PET TP2"
        @test length(MEH.tp_labels) == 3

        # TP indices sorted
        tp_indices = sort(collect(keys(MEH.tp_labels)))
        @test tp_indices == [0, 1, 2]
    end

    @testset "3. Main Event Channel" begin
        # main_event_channel is a Ref{Any}
        @test MEH.main_event_channel isa Ref

        # Can be set to nothing (no GUI)
        MEH.main_event_channel[] = nothing
        @test MEH.main_event_channel[] === nothing

        # Can be set to a Channel
        test_ch = Channel{Any}(10)
        MEH.main_event_channel[] = test_ch
        @test MEH.main_event_channel[] === test_ch

        # Channel event dispatch works
        put!(test_ch, SetTimePointEvent(1, 0))
        @test isready(test_ch)
        evt = take!(test_ch)
        @test evt isa SetTimePointEvent
        @test evt.tp_index == 1
        @test evt.panel == 0

        # Clean up
        close(test_ch)
        MEH.main_event_channel[] = nothing
    end

    @testset "4. LMW Observable Registry" begin
        # Verify the expected keys can be registered and accessed in LMW
        expected_keys = [:has_expert_edits, :seg_origin]
        for k in expected_keys
            @test haskey(LMW._lmw_observables, k)
        end

        # Test haskey Module extension
        for k in expected_keys
            @test haskey(LMW, k) == true
        end
    end

    @testset "5. LMW obs_next_lesion and obs_state Registration" begin
        # obs_next_lesion and obs_state are registered during GUI setup.
        # Without GUI, we can manually register and test the mechanism.
        test_obs = Observable(0)
        LMW._lmw_observables[:obs_next_lesion] = test_obs

        @test haskey(LMW._lmw_observables, :obs_next_lesion)
        @test LMW._lmw_observables[:obs_next_lesion] === test_obs

        next_lesion_called = Ref(false)
        on(test_obs) do _
            next_lesion_called[] = true
        end

        # Triggering the observable should fire the callback
        test_obs[] = test_obs[] + 1
        @test next_lesion_called[] == true

        # Clean up
        delete!(LMW._lmw_observables, :obs_next_lesion)
    end

    @testset "6. LMW obs_state Observable Mechanism" begin
        # Simulate obs_state registration
        obs_state = Observable("UNREVIEWED")
        LMW._lmw_observables[:obs_state] = obs_state

        @test haskey(LMW._lmw_observables, :obs_state)

        # Setting state via observable
        obs_state[] = "ACCEPTED"
        @test obs_state[] == "ACCEPTED"

        obs_state[] = "REJECTED"
        @test obs_state[] == "REJECTED"

        obs_state[] = "UNCERTAIN"
        @test obs_state[] == "UNCERTAIN"

        obs_state[] = "RESOLVED"
        @test obs_state[] == "RESOLVED"

        obs_state[] = "CORRECTED"
        @test obs_state[] == "CORRECTED"

        obs_state[] = "NEW"
        @test obs_state[] == "NEW"

        # Clean up
        delete!(LMW._lmw_observables, :obs_state)
    end

    @testset "7. Module haskey Extension Works" begin
        # Test Base.haskey extension for Module
        LMW._lmw_observables[:test_key] = Observable(42)
        @test haskey(LMW, :test_key) == true
        @test !haskey(LMW, :nonexistent_key)

        # Test Base.getindex extension
        @test LMW[:test_key] isa Observable
        @test LMW[:test_key][] == 42

        # Clean up
        delete!(LMW._lmw_observables, :test_key)
    end

    @testset "8. Event Struct Construction for Channel Dispatch" begin
        # These events are put on the channel by keyboard shortcuts
        @test AcceptLesionEvent() isa AcceptLesionEvent
        @test RejectLesionEvent() isa RejectLesionEvent
        @test MarkUncertainEvent() isa MarkUncertainEvent
        @test MarkResolvedEvent() isa MarkResolvedEvent

        # Test that they can be put on and taken from a Channel
        ch = Channel{Any}(10)
        put!(ch, AcceptLesionEvent())
        put!(ch, RejectLesionEvent())
        put!(ch, MarkUncertainEvent())
        put!(ch, MarkResolvedEvent())

        @test take!(ch) isa AcceptLesionEvent
        @test take!(ch) isa RejectLesionEvent
        @test take!(ch) isa MarkUncertainEvent
        @test take!(ch) isa MarkResolvedEvent

        close(ch)
    end

    # Reset state
    empty!(MEH.tp_labels)
    MEH.current_tp_index[] = 0
    MEH.main_event_channel[] = nothing
end
