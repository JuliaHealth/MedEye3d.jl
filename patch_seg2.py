import re

with open("src/display/GLFW/SegmentationDisplay.jl", "r") as f:
    code = f.read()

target = """                        push_consts[5] = ndc_left; push_consts[6] = ndc_bottom
                        push_consts[7] = ndc_right; push_consts[8] = ndc_top"""

new_code = """                        push_consts[5] = ndc_left; push_consts[6] = ndc_bottom
                        push_consts[7] = ndc_right; push_consts[8] = ndc_top
                        
                        ix, iy, iz = MakieEventHandlers.current_viewer_position[]
                        hovered_idx = MakieEventHandlers.current_hovered_panel[]
                        show_crosshair = 0.0f0
                        cx, cy = 0.0f0, 0.0f0
                        # Determine panel index from state object
                        # We are iterating over stateInstances. How to get panel_idx?
                        # wait, it is a for loop `for state in stateInstances`, but no index.
                        # Wait, we can get index from `findfirst(x -> x === state, stateInstances)`
                        # or better, use `for (panel_idx, state) in enumerate(stateInstances)`
"""

# Let's change `for state in stateInstances` to `for (panel_idx, state) in enumerate(stateInstances)`
code = code.replace("for state in stateInstances\n                        if state.calcDimsStruct", "for (panel_idx, state) in enumerate(stateInstances)\n                        if state.calcDimsStruct")

# Insert the crosshair logic
crosshair_logic = """                        push_consts[5] = ndc_left; push_consts[6] = ndc_bottom
                        push_consts[7] = ndc_right; push_consts[8] = ndc_top

                        ix, iy, iz = MakieEventHandlers.current_viewer_position[]
                        hovered_idx = MakieEventHandlers.current_hovered_panel[]
                        show_crosshair = 0.0f0
                        cx, cy = 0.0f0, 0.0f0

                        # Determine if crosshair toggle is active from LesionMetadataWindow
                        LMW = MakieEventHandlers._get_lmw()
                        is_crosshair_on = (LMW !== nothing) ? LMW.is_crosshair_visible() : false

                        if is_crosshair_on && ix > 0 && iy > 0 && iz > 0 && hovered_idx > 0 && panel_idx != hovered_idx
                            show_crosshair = 1.0f0
                            # Map 3D axial coordinate (ix, iy, iz) to panel's 2D texture (u, v)
                            dim_scroll = state.onScrollData.dimensionToScroll
                            w = Float32(state.calcDimsStruct.imageTextureWidth)
                            h = Float32(state.calcDimsStruct.imageTextureHeight)
                            if dim_scroll == 1 # Sagittal (Y, Z)
                                cx = iy / w
                                cy = iz / h
                            elseif dim_scroll == 2 # Coronal (X, Z)
                                cx = ix / w
                                cy = iz / h
                            else # Axial (X, Y)
                                cx = ix / w
                                cy = iy / h
                            end
                        end
                        push_consts[9] = cx; push_consts[10] = cy
                        push_consts[11] = show_crosshair; push_consts[12] = 0.0f0
"""

code = code.replace(target, crosshair_logic)

with open("src/display/GLFW/SegmentationDisplay.jl", "w") as f:
    f.write(code)
