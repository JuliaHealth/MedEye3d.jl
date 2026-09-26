path = "src/display/reactingToMouseKeyboard/ReactOnMouseClickAndDrag.jl"
content = read(path, String)

old_logic = """
    ix = round(Int, texX)
    iy = round(Int, texY)
    iz = panelState.currentDisplayedSlice
    
    if panelState.onScrollData.dimensionToScroll == 1
        iz = ix; ix = panelState.currentDisplayedSlice
    elseif panelState.onScrollData.dimensionToScroll == 2
        iz = iy; iy = panelState.currentDisplayedSlice
    end
    # Convert to primary axial (X, Y, Z) based on panel ID
    origX, origY, origZ = ix, iy, iz
    if clickedPanel == 3 # Sagittal (Y, Z, X) -> (X, Y, Z)
        origX, origY, origZ = iz, ix, iy
    elseif clickedPanel == 4 # Coronal (X, Z, Y) -> (X, Y, Z)
        origX, origY, origZ = ix, iz, iy
    end
"""

new_logic = """
    ix = round(Int, texX)
    iy = round(Int, texY)
    slice = panelState.currentDisplayedSlice
    
    if clickedPanel == 3 # Sagittal: X is slice, Y is texX, Z is texY
        origX = slice
        origY = ix
        origZ = iy
    elseif clickedPanel == 4 # Coronal: Y is slice, X is texX, Z is texY
        origX = ix
        origY = slice
        origZ = iy
    else # Axial: Z is slice, X is texX, Y is texY
        origX = ix
        origY = iy
        origZ = slice
    end
    
    # Clean up active (hover/drag) measurements from ALL OTHER panels to avoid ghost measurements
    for i in 1:length(mainStates)
        if i != clickedPanel
            filter!(m -> !m.is_active, mainStates[i].mainForDisplayObjects.measurements)
        end
    end
"""

content = replace(content, old_logic => new_logic)
write(path, content)
