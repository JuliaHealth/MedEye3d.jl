using MedEye3d
using MedEye3d.AppMain
using MedEye3d.SegmentationDisplay
using MedEye3d.ForDisplayStructs
using MedEye3d.MakieEvents

println("[TEST] Starting test_paint_new.jl with async runner...")
h5_path = "/workspaces/MedEye3d.jl/data/pat_6_files/preprocessed_volumes.h5"

@async begin
    try
        MEH = MedEye3d.SegmentationDisplay.MakieEventHandlers
        LMW = MedEye3d.LesionMetadataWindow
        
        println("[TEST] Waiting for _lmw_observables...")
        for _ in 1:60
            if haskey(LMW._lmw_observables, :active_lesion_id) && !MEH.app_is_loading[]
                break
            end
            sleep(1.0)
        end
        sleep(2.0)
        
        println("\n=== [DIAGNOSTIC TEST START] ===")
        println("Initial active_lesion_id: ", LMW._lmw_observables[:active_lesion_id][])
        println("Initial MEH.current_active_lesion_id: ", MEH.current_active_lesion_id[])
        println("Initial is_single_lesion_mode: ", MEH.is_single_lesion_mode[])

        states = MedEye3d.SegmentationDisplay.stateInstances
        println("State count: ", length(states))
        s1 = states[1]
        println("s1.valueForMasToSet before: ", s1.valueForMasToSet)
        println("s1.textureToModifyVec before: ", [t.name for t in s1.textureToModifyVec])
        for ts in s1.mainForDisplayObjects.listOfTextSpecifications
            if ts.name in ("Mask", "segmentation", "manualModif")
                println("  ts '$(ts.name)' before: isVisible=$(ts.isVisible), minMax=$(ts.minAndMaxValue), allowedIDs=$(ts.allowedIDs), maskContrib=$(ts.maskContribution)")
            end
        end

        # Click New Lesion
        println("\n>>> Clicking New Lesion (obs_new_lesion)...")
        LMW._lmw_observables[:obs_new_lesion][] = LMW._lmw_observables[:obs_new_lesion][] + 1
        sleep(2)

        println("\nAfter New Lesion:")
        println("active_lesion_id: ", LMW._lmw_observables[:active_lesion_id][])
        println("MEH.current_active_lesion_id: ", MEH.current_active_lesion_id[])
        println("MEH.is_single_lesion_mode: ", MEH.is_single_lesion_mode[])
        println("s1.valueForMasToSet: ", s1.valueForMasToSet)
        println("s1.textureToModifyVec: ", [t.name for t in s1.textureToModifyVec])
        for ts in s1.mainForDisplayObjects.listOfTextSpecifications
            if ts.name in ("Mask", "segmentation", "manualModif")
                println("  ts '$(ts.name)': isVisible=$(ts.isVisible), minMax=$(ts.minAndMaxValue), allowedIDs=$(ts.allowedIDs), maskContrib=$(ts.maskContribution)")
            end
        end

        # Now simulate mouse draw
        println("\n>>> Simulating mouse draw for new lesion...")
        println("s1.currentlyDispDat sliceNumber: ", s1.currentlyDispDat.sliceNumber)
        tex_name = s1.textureToModifyVec[1].name
        println("Target texture name: ", tex_name)
        println("has key in nameIndexes: ", haskey(s1.currentlyDispDat.nameIndexes, tex_name))
        if haskey(s1.currentlyDispDat.nameIndexes, tex_name)
            idx = s1.currentlyDispDat.nameIndexes[tex_name]
            twoDim = s1.currentlyDispDat.listOfDataAndImageNames[idx]
            println("twoDim.dat size: ", size(twoDim.dat), " eltype: ", eltype(twoDim.dat))
            println("nonzeros in twoDim.dat before: ", count(!iszero, twoDim.dat))
            
            # Simulate a mouse stroke
            mouseStruct = MouseStruct(
                isLeftButtonDown=true,
                lastCoordinates=[CartesianIndex(250, 250), CartesianIndex(251, 251), CartesianIndex(252, 252)],
                actualWindowWidth=1100,
                actualWindowHeight=1000
            )
            MedEye3d.ReactOnMouseClickAndDrag.react_to_draw([mouseStruct], s1, states)
            println("nonzeros in twoDim.dat after: ", count(!iszero, twoDim.dat))
            unique_vals = unique(twoDim.dat)
            println("unique values in twoDim.dat: ", unique_vals)
            
            # Check shader visibility formula for this value!
            paint_val = s1.valueForMasToSet.value
            ts_mask = nothing
            for ts in s1.mainForDisplayObjects.listOfTextSpecifications
                if ts.name == tex_name
                    ts_mask = ts
                    break
                end
            end
            if ts_mask !== nothing
                minV = ts_mask.minAndMaxValue[1]
                maxV = ts_mask.minAndMaxValue[2]
                println("Shader check for painted val=$paint_val:")
                println("  ts_mask.isVisible: ", ts_mask.isVisible)
                println("  ts_mask.maskContribution: ", ts_mask.maskContribution)
                println("  ts_mask.minAndMaxValue: ", [minV, maxV])
                println("  paint_val in range [minV, maxV]? ", (paint_val >= minV - 0.1 && paint_val <= maxV + 0.1))
                if !isempty(ts_mask.allowedIDs)
                    println("  allowedIDs: ", ts_mask.allowedIDs)
                    println("  paint_val in allowedIDs? ", any(x -> abs(paint_val - x) < 0.1, ts_mask.allowedIDs))
                end
            end
        end

        println("=== [DIAGNOSTIC TEST COMPLETE] ===")
    catch e
        println("[TEST ERROR] ", e)
        for (st_i, st_line) in enumerate(stacktrace(catch_backtrace()))
            println("  [$st_i] $st_line")
        end
    finally
        exit(0)
    end
end

MedEye3d.AppMain.launch_from_h5(h5_path; quad=false)
