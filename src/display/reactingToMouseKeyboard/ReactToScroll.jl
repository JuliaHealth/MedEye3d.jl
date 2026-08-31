
"""
module that holds functions needed to  react to scrolling
Generally first we need to pass the GLFW callback to the Rocket obeservable
code adapted from https://discourse.julialang.org/t/custom-subject-in-rocket-jl-for-mouse-events-from-glfw/65133/3
"""
module ReactToScroll
using GLFW, Logging
using ..DisplayWords, ..ForDisplayStructs, ..TextureManag, ..DataStructs, ..StructsManag, ..ShadersAndVerticiesForSupervoxels, ..ShadersAndVerticiesForLine, ..MakieEvents

export reactToScroll, reactToScrollZoom, reactToScrollMultiPanel!
export registerMouseScrollFunctions

# Module-local PET/CT blend tracking for Ctrl+scroll
const _pet_blend_ref = Ref(1.0f0)




"""
uploading data to given texture; of given types associated
returns subscription in order to enable unsubscribing in the end
window - GLFW window
return scrollback - that holds boolean subject (observable) to which we can react by subscribing appropriate actor
"""
function registerMouseScrollFunctions(window::GLFW.Window, mainChannel::Base.Channel{Any})
    GLFW.SetScrollCallback(window, (a, xoff, yoff) -> begin
        ctrl_down = GLFW.GetKey(window, GLFW.KEY_LEFT_CONTROL) == GLFW.PRESS || GLFW.GetKey(window, GLFW.KEY_RIGHT_CONTROL) == GLFW.PRESS
        shift_down = GLFW.GetKey(window, GLFW.KEY_LEFT_SHIFT) == GLFW.PRESS || GLFW.GetKey(window, GLFW.KEY_RIGHT_SHIFT) == GLFW.PRESS
        alt_down = GLFW.GetKey(window, GLFW.KEY_LEFT_ALT) == GLFW.PRESS || GLFW.GetKey(window, GLFW.KEY_RIGHT_ALT) == GLFW.PRESS
        
        # Ensure switchIndex is updated to the quadrant directly under the cursor
        try
            actualW, actualH = GLFW.GetWindowSize(window)
            curX, curY = GLFW.GetCursorPos(window)
            if curX >= 0 && curX <= actualW && curY >= 0 && curY <= actualH
                put!(mainChannel, MouseStruct(
                    isLeftButtonDown  = false,
                    isRightButtonDown = false,
                    lastCoordinates   = [CartesianIndex(Int(round(curX)), Int(round(curY)))],
                    actualWindowWidth  = Int(actualW),
                    actualWindowHeight = Int(actualH),
                ))
            end
        catch
        end
        
        if ctrl_down && !shift_down && !alt_down
            # Ctrl+scroll: adjust PET/CT blend (±0.05 per tick)
            _pet_blend_ref[] = clamp(_pet_blend_ref[] + Float32(yoff > 0 ? 0.05 : -0.05), 0.0f0, 1.0f0)
            put!(mainChannel, PetBlendEvent(_pet_blend_ref[]))
        elseif shift_down || alt_down
            put!(mainChannel, ScrollZoomEvent(Float64(yoff)))
        else
            scroll_delta = yoff > 0 ? 1 : (yoff < 0 ? -1 : 0)
            if scroll_delta != 0
                put!(mainChannel, Int64(scroll_delta))
            end
        end
    end)
end #registerMouseScrollFunctions

"""
    reactToScrollZoom(data::ScrollZoomEvent, mainStates::Vector{StateDataFields})

Handles continuous scaling logic when holding `Shift` + `Scroll`.
The zooming dynamically recalculates `calcDimsStruct.zoom` and clips bounds (1.0x - 20.0x zoom). Automatically resets panning logic when fully zoomed out. Triggers a render pass immediately upon recalculation.
"""
function reactToScrollZoom(data::ScrollZoomEvent, mainStates::Vector{StateDataFields})
    panelIdx = mainStates[1].switchIndex
    if panelIdx < 1 || panelIdx > length(mainStates)
        panelIdx = 1
    end
    mainState = mainStates[panelIdx]
    
    # Zoom factor: 1.1 for each scroll tick
    zoomFactor = data.zoom_delta > 0 ? 1.1f0 : (1.0f0 / 1.1f0)
    
    # Increase zoom, clamp between 1.0 (no zoom) and 20.0
    newZoom = clamp(mainState.calcDimsStruct.zoom * zoomFactor, 1.0f0, 20.0f0)
    
    # If zoom hits 1.0, reset panning as well to keep it clean
    if newZoom == 1.0f0
        mainState.calcDimsStruct.panX = 0.0f0
        mainState.calcDimsStruct.panY = 0.0f0
    end
    
    mainState.calcDimsStruct.zoom = newZoom
    @info "Shift-Scroll Zoom: $(round(newZoom, digits=2))x (panel=$panelIdx)"
    # GPU zoom: no reactToScroll needed — render loop picks up new zoom via setZoomPanUniforms
end



"""
    reactToScroll(scrollNumb::Int64, mainStates::Vector{StateDataFields}, toBeSavedForBack::Bool=true)

Core scroll-navigation function executed sequentially by the `GL_Consumer` thread.

# Logic Flow:
1. Calculates target slice by applying `scrollNumb` (multiplied by 10 if fast-scroll `Shift` is active).
2. Extracts the 2D cross-section data (`SingleSliceDat`) from the 3D raw voxel volume (`ThreeDimRawDat`) associated with the active panel.
3. Automatically triggers 3D synchronizations (`reactToScroll` loops recursively) for QuadView configurations. Updates orthogonal coronal/sagittal panels immediately based on world-coordinate intersections with the axial view.
4. Uploads data into GPU textures and calls `glClear`/`glDrawElements` via the consumer block.
"""
function reactToScroll(scrollNumb::Int64, mainStates::Vector{StateDataFields}, toBeSavedForBack::Bool=true)
    t_start = time_ns()
    clickedPanel = mainStates[1].switchIndex
    if clickedPanel < 1 || clickedPanel > length(mainStates)
        clickedPanel = 1
    end
    mainState = mainStates[clickedPanel]

    delta = mainState.mainForDisplayObjects.isFastScroll ? scrollNumb * 10 : scrollNumb

    if length(mainStates) > 1 && mainState.mainForDisplayObjects.isSyncScrollOn
        active_panels = Int[]
        slice_overrides = Dict{Int,Int}()
        for (i, pState) in enumerate(mainStates)
            lastSlice = pState.onScrollData.slicesNumber
            if lastSlice > 0
                push!(active_panels, i)
                newSlice = clamp(pState.currentDisplayedSlice + delta, 1, lastSlice)
                slice_overrides[i] = newSlice
            end
        end
        reactToScrollMultiPanel!(active_panels, mainStates, slice_overrides)
    else
        # Single panel scroll
        lastSlice = mainState.onScrollData.slicesNumber
        if lastSlice > 0
            newSlice = clamp(mainState.currentDisplayedSlice + delta, 1, lastSlice)
            reactToScrollMultiPanel!([clickedPanel], mainStates, Dict(clickedPanel => newSlice))
        end
    end

    t_end = time_ns()
    action = scrollNumb != 0 ? "SCROLL" : "REDRAW"
    @info "[BENCH] reactToScroll ($(action)): $(round((t_end-t_start)/1e6, digits=1))ms"
end#reactToScroll
function getSvCurrentSlice(all_supervoxels::Dict{Int, Dict{Int, Dict{String,Any}}}, slice_number, mainState=nothing)
    # Get the current axis
    current_axis = mainState !== nothing ? mainState.onScrollData.dimensionToScroll : 3

    # Check if we have supervoxels for this axis
    if !haskey(all_supervoxels, current_axis)
        return Dict{String,Any}(
            "supervoxel_vertices" => Float32[],
            "supervoxel_indices" => UInt32[],
            "slice_position" => Float64(slice_number)
        )
    end

    # Get the supervoxels for the current axis
    axis_supervoxels = all_supervoxels[current_axis]

    # Find the closest slice
    slice_positions = [sv["slice_position"] for (_, sv) in axis_supervoxels]
    if isempty(slice_positions)
        return Dict{String,Any}(
            "supervoxel_vertices" => Float32[],
            "supervoxel_indices" => UInt32[],
            "slice_position" => Float64(slice_number)
        )
    end

    closest_slice_key = argmin(abs.(collect(keys(axis_supervoxels)) .- slice_number))
    slice_key = collect(keys(axis_supervoxels))[closest_slice_key]

    return axis_supervoxels[slice_key]
end

# function getSvCurrentSlice(all_supervoxels::Dict{Int, Dict{String,Any}}, slice_number)
#     if haskey(all_supervoxels, slice_number)
#         return all_supervoxels[slice_number]
#     else
#         return Dict{String,Any}(
#             "supervoxel_vertices" => Float32[],
#             "supervoxel_indices" => UInt32[],
#             "slice_position" => Float64(slice_number)
#             )

#     end
# end

"""
    reactToScrollMultiPanel!(panels, mainStates, sliceOverrides)

Batch texture upload for multiple panels in ONE pass. Only uploads texture data
(glTexSubImage2D) without text rendering or draw calls — consumer loop renders.

This replaces N sequential `reactToScroll(0, ...)` calls with a single batch,
avoiding per-panel overhead:
  - No per-panel glUseProgram / glBufferData for text shader
  - No per-panel FreeType CPU text rendering
  - No per-panel glDrawElements
  - No per-panel reactivateMainObj buffer upload

`sliceOverrides` is a Dict{Int,Int} mapping panel_idx => slice_number.
If a panel is not in sliceOverrides, its currentDisplayedSlice is used.
"""
function reactToScrollMultiPanel!(panels::Vector{Int}, mainStates::Vector{StateDataFields},
                                  sliceOverrides::Dict{Int,Int}=Dict{Int,Int}())
    t_start = time_ns()
    n_uploads = 0
    for panel_idx in panels
        if panel_idx < 1 || panel_idx > length(mainStates)
            continue
        end
        panelState = mainStates[panel_idx]
        lastSlice = panelState.onScrollData.slicesNumber
        if lastSlice < 1
            continue
        end
        
        prev_slice = panelState.currentDisplayedSlice
        current = get(sliceOverrides, panel_idx, prev_slice)
        current = clamp(current, 1, lastSlice)
        slice_changed = (current != prev_slice) || (panelState.currentlyDispDat.sliceNumber == 0)
        panelState.currentDisplayedSlice = current
        panelState.isSliceChanged = slice_changed
        
        # Slice 3D→2D for all textures in this panel
        singleSlDat = panelState.onScrollData.dataToScroll |>
            (scrDat) -> map(threeDimDat -> threeToTwoDimm(threeDimDat.type, Int64(current), panelState.onScrollData.dimensionToScroll, threeDimDat), scrDat) |>
            (twoDimList) -> SingleSliceDat(listOfDataAndImageNames=twoDimList, sliceNumber=current)
        
        # Upload image textures only (no text, no draw calls)
        modulelistOfTextSpecs = panelState.mainForDisplayObjects.listOfTextSpecifications
        calcDimStruct = panelState.calcDimsStruct
        for updateDat in singleSlDat.listOfDataAndImageNames
            # Optimization: if slice didn't change, only upload dynamic overlays (Bone, Mask)
            if !slice_changed && updateDat.name != "Bone_Surface" && updateDat.name != "Bone_Marrow" && updateDat.name != "Mask" && updateDat.name != "manualModif"
                continue
            end
            findList = findall((texSpec) -> texSpec.name == updateDat.name, modulelistOfTextSpecs)
            if !isempty(findList)
                texSpec = modulelistOfTextSpecs[findList[1]]
                # GPU zoom/pan: upload raw unzoomed data — zoom/pan applied by vertex shader uvScale/uvOffset uniforms
                TextureManag.updateTexture(updateDat.type, updateDat.dat, texSpec, 0, 0, calcDimStruct.imageTextureWidth, calcDimStruct.imageTextureHeight)
                n_uploads += 1
            end
        end
        
        # Upload text texture ONLY for panels with active text display (typically panel 1)
        if (panel_idx == 1 || panelState.textDispObj.textureSpec.ID[] != 0) && calcDimStruct.wordsQuadVertSize > 0 && !all(calcDimStruct.wordsImageQuadVert .== 0.0f0)
            DisplayWords.activateForTextDisp(panelState.textDispObj.shader_program_words, panelState.textDispObj.vbo_words, calcDimStruct)
            TextureManag.addTextToTexture(panelState.textDispObj, [singleSlDat.textToDisp..., panelState.valueForMasToSet.text], calcDimStruct)
            DisplayWords.reactivateMainObj(panelState.mainForDisplayObjects.shader_program, panelState.mainForDisplayObjects.vbo, calcDimStruct)
        end
        
        panelState.currentlyDispDat = singleSlDat
        
        # Update mouse position tracking
        currentDim = Int64(panelState.onScrollData.dataToScrollDims.dimensionToScroll)
        lastMouse = panelState.lastRecordedMousePosition
        locArr = [lastMouse[1], lastMouse[2], lastMouse[3]]
        locArr[3] = current
        panelState.lastRecordedMousePosition = CartesianIndex(locArr[1], locArr[2], locArr[3])
    end
    t_ms = (time_ns() - t_start) / 1e6
    if t_ms > 20.0
        println("  [BENCH] reactToScrollMultiPanel!($(length(panels)) panels, $n_uploads tex): $(round(t_ms, digits=1))ms"); flush(stdout)
    end
end

end #ReactToScroll
