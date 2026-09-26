path = "src/display/reactingToMouseKeyboard/ReactOnMouseClickAndDrag.jl"
content = read(path, String)

# Remove the two bad injections
bad_str = """
    # Determine measurement sub-mode
    MEH = parentmodule(@__MODULE__).SegmentationDisplay.MakieEventHandlers
    sub_mode = MEH.measurement_sub_mode[]
    MeasMod = parentmodule(@__MODULE__).Measurements
    
    if sub_mode == :sphere
        _handle_sphere_measurement(mousestr, mainStates, panelState.mainForDisplayObjects, (Float32(origX), Float32(origY), Float32(origZ)), MEH, MeasMod)
    elseif sub_mode == :line
        _handle_line_measurement(mousestr, mainStates, panelState.mainForDisplayObjects, (Float32(origX), Float32(origY), Float32(origZ)), MEH, MeasMod)
    end
    
    if panelState.moveLesionMode
"""
content = replace(content, bad_str => "    if panelState.moveLesionMode\n")
write(path, content)
