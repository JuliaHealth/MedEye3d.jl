path = "src/display/reactingToMouseKeyboard/ReactOnMouseClickAndDrag.jl"
content = read(path, String)

react_to_draw_inject = """
    stateObject = mainStates[mainStates[1].switchIndex]
    
    # --- MEASUREMENT INTERCEPT (DRAG) ---
    MEH = parentmodule(@__MODULE__).SegmentationDisplay.MakieEventHandlers
    sub_mode = MEH.measurement_sub_mode[]
    if sub_mode == :sphere || sub_mode == :line
        MeasMod = parentmodule(@__MODULE__).Measurements
        calcDim = stateObject.calcDimsStruct
        first_mouse = mouseStructArray[end]
        actualW = first_mouse.actualWindowWidth > 0 ? Float64(first_mouse.actualWindowWidth) : Float64(calcDim.windowWidth)
        actualH = first_mouse.actualWindowHeight > 0 ? Float64(first_mouse.actualWindowHeight) : Float64(calcDim.windowHeight)
        
        c = first_mouse.lastCoordinates[1]
        texX, texY = StructsManag.getTextureCoordinatesFromScreen(c[1], c[2], calcDim, actualW, actualH)
        
        ix, iy = round(Int, texX), round(Int, texY)
        clickedPanel = mainStates[1].switchIndex
        slice = stateObject.currentDisplayedSlice
        
        cp_mapped = clickedPanel > 5 ? clickedPanel - 5 : clickedPanel
        if cp_mapped == 3 # Sagittal
            origX, origY, origZ = slice, ix, iy
        elseif cp_mapped == 4 # Coronal
            origX, origY, origZ = ix, slice, iy
        else # Axial (1,2,5)
            origX, origY, origZ = ix, iy, slice
        end
        
        # Clean up active ghost measurements from other panels
        for i in 1:length(mainStates)
            if i != clickedPanel
                filter!(m -> !m.is_active, mainStates[i].mainForDisplayObjects.measurements)
                filter!(m -> !m.is_active, mainStates[i].mainForDisplayObjects.line_measurements)
            end
        end
        
        if sub_mode == :sphere
            _handle_sphere_measurement(first_mouse, mainStates, stateObject.mainForDisplayObjects, (Float32(origX), Float32(origY), Float32(origZ)), MEH, MeasMod)
        elseif sub_mode == :line
            _handle_line_measurement(first_mouse, mainStates, stateObject.mainForDisplayObjects, (Float32(origX), Float32(origY), Float32(origZ)), MEH, MeasMod)
        end
        return # Do not paint
    end
    # --- END MEASUREMENT INTERCEPT ---

    if !stateObject.valueForMasToSet.is_painting_active || isempty(stateObject.textureToModifyVec)
"""
content = replace(content, """
    stateObject = mainStates[mainStates[1].switchIndex]
    if !stateObject.valueForMasToSet.is_painting_active || isempty(stateObject.textureToModifyVec)
""" => react_to_draw_inject)


react_to_mouse_drag_inject = """
    # If the left mouse button is released, clear the paint stroke tail
    if !mousestr.isLeftButtonDown
        for state in mainStates
            empty!(state.lastPaintCoords)
        end
        
        # --- MEASUREMENT INTERCEPT (RELEASE) ---
        MEH = parentmodule(@__MODULE__).SegmentationDisplay.MakieEventHandlers
        sub_mode = MEH.measurement_sub_mode[]
        if sub_mode == :sphere || sub_mode == :line
            clickedPanel = mainState.switchIndex
            if clickedPanel >= 1 && clickedPanel <= length(mainStates)
                stateObject = mainStates[clickedPanel]
                MeasMod = parentmodule(@__MODULE__).Measurements
                calcDim = stateObject.calcDimsStruct
                actualW = mousestr.actualWindowWidth > 0 ? Float64(mousestr.actualWindowWidth) : Float64(calcDim.windowWidth)
                actualH = mousestr.actualWindowHeight > 0 ? Float64(mousestr.actualWindowHeight) : Float64(calcDim.windowHeight)
                
                if !isempty(mouseCoords)
                    c = mouseCoords[1]
                    texX, texY = StructsManag.getTextureCoordinatesFromScreen(c[1], c[2], calcDim, actualW, actualH)
                    ix, iy = round(Int, texX), round(Int, texY)
                    slice = stateObject.currentDisplayedSlice
                    
                    cp_mapped = clickedPanel > 5 ? clickedPanel - 5 : clickedPanel
                    if cp_mapped == 3 # Sagittal
                        origX, origY, origZ = slice, ix, iy
                    elseif cp_mapped == 4 # Coronal
                        origX, origY, origZ = ix, slice, iy
                    else # Axial
                        origX, origY, origZ = ix, iy, slice
                    end
                    
                    if sub_mode == :sphere
                        _handle_sphere_measurement(mousestr, mainStates, stateObject.mainForDisplayObjects, (Float32(origX), Float32(origY), Float32(origZ)), MEH, MeasMod)
                    elseif sub_mode == :line
                        _handle_line_measurement(mousestr, mainStates, stateObject.mainForDisplayObjects, (Float32(origX), Float32(origY), Float32(origZ)), MEH, MeasMod)
                    end
                end
            end
        end
        # --- END MEASUREMENT INTERCEPT ---
    end
"""
content = replace(content, """
    # If the left mouse button is released, clear the paint stroke tail
    if !mousestr.isLeftButtonDown
        for state in mainStates
            empty!(state.lastPaintCoords)
        end
    end
""" => react_to_mouse_drag_inject)

write(path, content)
println("Injected properly!")
