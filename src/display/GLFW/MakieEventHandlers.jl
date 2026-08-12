module MakieEventHandlers
using ...MakieEvents
using ...StructsManag
using ...ForDisplayStructs
using ...DataStructs
using ...ChangePlane
using ...ReactToScroll
using DataTypesBasic
using Setfield

export reactToChangePlane, reactToCompareTimePoints, reactToShowSingleLesion
export reactToWindowing, reactToPaintVal, reactToSyncLesion
export reactToChangeTimePoint, reactToToggleLesion, reactToRefreshList
export reactToAddAutoPet, reactToSyncMissing, reactToGenManual
export reactToMapLink, reactToAutoRunPreprocess, reactToRunPreprocess, reactToShowBoneMask, reactToSaveMRB
using ...InferenceClient
using ...LesionAssociation
using ModernGL
using ..Uniforms
function reactToChangePlane(data::ChangePlaneEvent, stateObjects::Vector{StateDataFields})
    dim = 3
    if data.plane == :Sagittal
        dim = 1
    elseif data.plane == :Coronal
        dim = 2
    elseif data.plane == :Axial
        dim = 3
    end
    
    dummy_kb = KeyboardStruct()
    old_idx = stateObjects[1].switchIndex
    for (idx, stateObject) in enumerate(stateObjects)
        old_scroll = stateObject.onScrollData.dataToScrollDims
        new_scroll = DataToScrollDims(imageSize=old_scroll.imageSize, voxelSize=old_scroll.voxelSize, dimensionToScroll=dim)
        
        # Override the lastRecordedMousePosition to be the middle of the volume
        # so that ChangePlane.processKeysInfo will extract the middle slice instead of slice 1 (which is usually black air)
        stateObject.lastRecordedMousePosition = CartesianIndex(
            max(1, round(Int, old_scroll.imageSize[1] / 2)),
            max(1, round(Int, old_scroll.imageSize[2] / 2)),
            max(1, round(Int, old_scroll.imageSize[3] / 2))
        )
        
        ChangePlane.processKeysInfo(Identity(new_scroll), stateObject, dummy_kb, false)
        
        stateObjects[1].switchIndex = idx
        ReactToScroll.reactToScroll(0, stateObjects, false)
    end
    stateObjects[1].switchIndex = old_idx
end

function updateQuadVertices!(stateObject::StateDataFields, layout::Symbol)
    calcDimStruct = stateObject.calcDimsStruct
    res = copy(calcDimStruct.mainImageQuadVert)
    
    if layout == :Hidden
        res .= 0.0f0
    else
        # Recalculate aspect ratio corrections based on the active layout
        ratio_desired = calcDimStruct.heightToWithRatio * (calcDimStruct.imageTextureHeight / calcDimStruct.imageTextureWidth)
        
        av_width = Float64(calcDimStruct.windowWidth)
        av_height = Float64(calcDimStruct.windowHeight)
        
        if layout == :LeftHalf || layout == :RightHalf
            av_width /= 2.0
        elseif layout == :TopLeft || layout == :TopRight || layout == :BottomLeft || layout == :BottomRight
            av_width /= 2.0
            av_height /= 2.0
        end
        
        ratio_actual = av_height / av_width
        
        widthCorr = 0.0f0
        heightCorr = 0.0f0
        if ratio_actual > ratio_desired
            heightCorr = 1.0f0 - Float32(ratio_desired / ratio_actual)
        else
            widthCorr = 1.0f0 - Float32(ratio_actual / ratio_desired)
        end
        
        # OpenGL width is 1.0 for all these layouts
        wc = widthCorr / 2.0f0
        
        # OpenGL height is 2.0 for LeftHalf/RightHalf, and 1.0 for quadrants
        hc = (layout == :LeftHalf || layout == :RightHalf) ? heightCorr : heightCorr / 2.0f0
        
        scale = 0.90f0
        yOffset = -0.10f0
        
        # Y coordinates
        if layout == :TopLeft || layout == :TopRight
            res[2] = (1.0f0 - hc) * scale + yOffset
            res[10] = (0.0f0 + hc) * scale + yOffset
            res[18] = (0.0f0 + hc) * scale + yOffset
            res[26] = (1.0f0 - hc) * scale + yOffset
        elseif layout == :BottomLeft || layout == :BottomRight
            res[2] = (0.0f0 - hc) * scale + yOffset
            res[10] = (-1.0f0 + hc) * scale + yOffset
            res[18] = (-1.0f0 + hc) * scale + yOffset
            res[26] = (0.0f0 - hc) * scale + yOffset
        elseif layout == :LeftHalf || layout == :RightHalf
            res[2] = (1.0f0 - hc) * scale + yOffset
            res[10] = (-1.0f0 + hc) * scale + yOffset
            res[18] = (-1.0f0 + hc) * scale + yOffset
            res[26] = (1.0f0 - hc) * scale + yOffset
        end
        
        # X coordinates
        if layout == :TopLeft || layout == :BottomLeft || layout == :LeftHalf
            res[1] = (0.0f0 - wc) * scale
            res[9] = (0.0f0 - wc) * scale
            res[17] = (-1.0f0 + wc) * scale
            res[25] = (-1.0f0 + wc) * scale
        elseif layout == :TopRight || layout == :BottomRight || layout == :RightHalf
            res[1] = (1.0f0 - wc) * scale
            res[9] = (1.0f0 - wc) * scale
            res[17] = (0.0f0 + wc) * scale
            res[25] = (0.0f0 + wc) * scale
        end
    end
    
    stateObject.calcDimsStruct = Setfield.setproperties(calcDimStruct, (mainImageQuadVert = res,))
    
    ModernGL.glBindBuffer(ModernGL.GL_ARRAY_BUFFER, stateObject.mainForDisplayObjects.vbo)
    ModernGL.glBufferData(ModernGL.GL_ARRAY_BUFFER, sizeof(res), res, ModernGL.GL_STATIC_DRAW)
end

const compare_mode = Ref(false)
const compare_right_tp = Ref(-1)  # TP index shown in right panel (panel 5)

function reactToCompareTimePoints(data::CompareTimePointsEvent, stateObjects::Vector{StateDataFields})
    if length(stateObjects) >= 5
        compare_mode[] = data.compare
        if data.compare
            # 2-pane view: panel 1 on left, panel 5 on right
            updateQuadVertices!(stateObjects[1], :LeftHalf)
            updateQuadVertices!(stateObjects[5], :RightHalf)
            updateQuadVertices!(stateObjects[2], :Hidden)
            updateQuadVertices!(stateObjects[3], :Hidden)
            updateQuadVertices!(stateObjects[4], :Hidden)
            
            # Load the NEXT TP into panel 5
            if !isempty(tp_data_cache)
                tp_indices = sort(collect(keys(tp_data_cache)))
                cur_pos = findfirst(==(current_tp_index[]), tp_indices)
                if cur_pos === nothing
                    cur_pos = 1
                end
                # Right panel shows the next TP chronologically
                next_pos = mod1(cur_pos + 1, length(tp_indices))
                right_tp = tp_indices[next_pos]
                compare_right_tp[] = right_tp
                
                # Load right TP data into panel 5
                tp_voxels = tp_data_cache[right_tp]
                
                # Ensure manualModif is inserted at index 2 if it's missing (to match SegmentationDisplay.jl initialization)
                if length(tp_voxels) >= 1 && tp_voxels[1][1] != "manualModif" && (length(tp_voxels) < 2 || tp_voxels[2][1] != "manualModif")
                    insert!(tp_voxels, 2, ("manualModif", zeros(Float32, size(tp_voxels[1][2]))))
                end

                if length(tp_voxels) >= 5
                    newDataToScroll = StructsManag.getThreeDims(tp_voxels[5])
                    stateObjects[5].onScrollData.dataToScroll = newDataToScroll
                    stateObjects[5].onScrollData.nameIndexes = DataStructs.getLocationDict(newDataToScroll)
                    dimToScroll = stateObjects[5].onScrollData.dimensionToScroll
                    if !isempty(newDataToScroll)
                        stateObjects[5].onScrollData.slicesNumber = Int32(size(newDataToScroll[1].dat, dimToScroll))
                    end
                    stateObjects[5].currentDisplayedSlice = max(1, stateObjects[5].onScrollData.slicesNumber ÷ 2)
                elseif length(tp_voxels) >= 1
                    # Panel 5 uses same view as panel 1 (axial), so use index 1
                    newDataToScroll = StructsManag.getThreeDims(tp_voxels[1])
                    stateObjects[5].onScrollData.dataToScroll = newDataToScroll
                    stateObjects[5].onScrollData.nameIndexes = DataStructs.getLocationDict(newDataToScroll)
                    dimToScroll = stateObjects[5].onScrollData.dimensionToScroll
                    if !isempty(newDataToScroll)
                        stateObjects[5].onScrollData.slicesNumber = Int32(size(newDataToScroll[1].dat, dimToScroll))
                    end
                    stateObjects[5].currentDisplayedSlice = max(1, stateObjects[5].onScrollData.slicesNumber ÷ 2)
                end
                
                # Re-render both panels
                old_idx = stateObjects[1].switchIndex
                stateObjects[1].switchIndex = 1
                ReactToScroll.reactToScroll(0, stateObjects, false)
                stateObjects[1].switchIndex = 5
                ReactToScroll.reactToScroll(0, stateObjects, false)
                stateObjects[1].switchIndex = old_idx
                
                left_label = get(tp_labels, current_tp_index[], "TP $(current_tp_index[])")
                right_label = get(tp_labels, right_tp, "TP $right_tp")
                @info "Compare mode ON: Left=$left_label, Right=$right_label"
            end
        else
            compare_right_tp[] = -1
            # 4-pane view
            updateQuadVertices!(stateObjects[1], :TopLeft)
            updateQuadVertices!(stateObjects[2], :TopRight)
            updateQuadVertices!(stateObjects[3], :BottomLeft)
            updateQuadVertices!(stateObjects[4], :BottomRight)
            updateQuadVertices!(stateObjects[5], :Hidden)
            @info "Compare mode OFF: restored 4-pane view"
        end
    end
end

function reactToShowSingleLesion(data::ShowSingleLesionEvent, stateObjects::Vector{StateDataFields})
    changed = false
    for stateObject in stateObjects
        for textSpec in stateObject.mainForDisplayObjects.listOfTextSpecifications
            if textSpec.isMultiDiscreteMask || textSpec.name == "Mask"
                if data.lesion_id == 0
                    textSpec.minAndMaxValue = Float32.([1.0, 1000.0])
                else
                    textSpec.minAndMaxValue = Float32.([data.lesion_id, data.lesion_id])
                end
                
                # Push uniform update for min/max
                ModernGL.glUseProgram(stateObject.mainForDisplayObjects.shader_program)
                Uniforms.coontrolMinMaxUniformVals(textSpec)
                changed = true
            end
        end
    end
    @info "Show single lesion: $(data.lesion_id == 0 ? "all" : string(data.lesion_id))"
    return changed
end

function reactToWindowing(data::WindowingEvent, stateObjects::Vector{StateDataFields})
    for state in stateObjects
        for tex in state.mainForDisplayObjects.listOfTextSpecifications
            if tex.isMainImage
                tex.minAndMaxValue = Float32.([data.min_val, data.max_val])
                
                # Push uniform update for min/max
                ModernGL.glUseProgram(state.mainForDisplayObjects.shader_program)
                Uniforms.coontrolMinMaxUniformVals(tex)
            end
        end
    end
end

function reactToPaintVal(data::PaintValEvent, stateObjects::Vector{StateDataFields})
    for state in stateObjects
        state.valueForMasToSet.value = data.val
    end
end

function reactToSyncLesion(data::SyncLesionEvent, stateObjects::Vector{StateDataFields})
    changed = false
    old_idx = stateObjects[1].switchIndex
    old_sync = stateObjects[1].mainForDisplayObjects.isSyncScrollOn
    for stateObject in stateObjects
        stateObject.mainForDisplayObjects.isSyncScrollOn = false
        
        # NEW: Toggle single lesion visibility mask
        for textSpec in stateObject.mainForDisplayObjects.listOfTextSpecifications
            if textSpec.isMultiDiscreteMask || textSpec.name == "Mask"
                if data.lesion_id > 0
                    textSpec.minAndMaxValue = Float32.([data.lesion_id, data.lesion_id])
                else
                    textSpec.minAndMaxValue = Float32.([1.0, 1000.0])
                end
                ModernGL.glUseProgram(stateObject.mainForDisplayObjects.shader_program)
                Uniforms.coontrolMinMaxUniformVals(textSpec)
            end
        end
    end
    for (idx, stateObject) in enumerate(stateObjects)
        if data.lesion_id > 0
            found_center = nothing
            for (scrIdx, scrDat) in enumerate(stateObject.onScrollData.dataToScroll)
                texSpec = stateObject.mainForDisplayObjects.listOfTextSpecifications[scrIdx]
                if texSpec.name == "Mask" || texSpec.name == "manualModif"
                    # Using isapprox for Float32 comparison since mask might be float
                    indices = findall(x -> isapprox(x, Float32(data.lesion_id), atol=0.1f0), scrDat.dat)
                    if !isempty(indices)
                        # Extract the 3D mask matching the lesion
                        mask = isapprox.(scrDat.dat, Float32(data.lesion_id), atol=0.1f0)
                        
                        # Sum over axes to find the slice with the maximum area
                        best_x = argmax(sum(mask, dims=(2, 3)))[1]
                        best_y = argmax(sum(mask, dims=(1, 3)))[2]
                        best_z = argmax(sum(mask, dims=(1, 2)))[3]
                        
                        found_center = [best_x, best_y, best_z]
                        break
                    end
                end
            end
            
            if found_center !== nothing
                stateObject.lastRecordedMousePosition = CartesianIndex(found_center[1], found_center[2], found_center[3])
                dim = stateObject.onScrollData.dimensionToScroll
                stateObject.currentDisplayedSlice = found_center[dim]
                
                stateObjects[1].switchIndex = idx
                ReactToScroll.reactToScroll(0, stateObjects, false)
                changed = true
                @info "Synced active lesion $(data.lesion_id) at center $found_center in window $idx"
            else
                if idx > 1 && length(stateObjects) > 0
                    tp0_state = stateObjects[1]
                    pos_tp0 = tp0_state.lastRecordedMousePosition
                    spacing_tp0 = tp0_state.spacingsValue[1]
                    origin_tp0 = tp0_state.originValue[1]
                    phys_x = origin_tp0[1] + (pos_tp0[1] - 1) * spacing_tp0[1]
                    phys_y = origin_tp0[2] + (pos_tp0[2] - 1) * spacing_tp0[2]
                    phys_z = origin_tp0[3] + (pos_tp0[3] - 1) * spacing_tp0[3]
                    
                    spacing_tp1 = stateObject.spacingsValue[1]
                    origin_tp1 = stateObject.originValue[1]
                    vox_x = round(Int, (phys_x - origin_tp1[1]) / spacing_tp1[1]) + 1
                    vox_y = round(Int, (phys_y - origin_tp1[2]) / spacing_tp1[2]) + 1
                    vox_z = round(Int, (phys_z - origin_tp1[3]) / spacing_tp1[3]) + 1
                    
                    mapped_pos = CartesianIndex(vox_x, vox_y, vox_z)
                    if idx == 1 || idx == 2
                        stateObject.lastRecordedMousePosition = mapped_pos
                    elseif idx == 3
                        stateObject.lastRecordedMousePosition = CartesianIndex(vox_y, vox_z, vox_x)
                    else
                        stateObject.lastRecordedMousePosition = CartesianIndex(vox_x, vox_z, vox_y)
                    end
                    
                    dim = stateObject.onScrollData.dimensionToScroll
                    stateObject.currentDisplayedSlice = mapped_pos[dim]
                    
                    stateObjects[1].switchIndex = idx
                    ReactToScroll.reactToScroll(0, stateObjects, false)
                    changed = true
                    @info "Mapped lesion $(data.lesion_id) via physical coordinates to window $idx: $mapped_pos"
                end
            end
        end
    end
    for stateObject in stateObjects
        stateObject.mainForDisplayObjects.isSyncScrollOn = old_sync
    end
    stateObjects[1].switchIndex = old_idx
    return changed
end

# TP navigation state: populated by the launch script
# tp_data_cache[tp_index] = Vector{Vector{Any}} — per-panel voxel data tuples [(\"CT\", vol), (\"PET\", vol), (\"Mask\", vol)]
const tp_data_cache = Dict{Int, Vector{Vector{Any}}}()
const current_tp_index = Ref(0)
const tp_labels = Dict{Int, String}()  # tp_index → display label (e.g. "PET TP0")
export tp_data_cache, current_tp_index, tp_labels
export compare_mode, compare_right_tp

"""
Helper: load TP data into a specific panel's onScrollData and re-render.
`panel_idx` is the stateObjects index (1-5).
`tp_voxel_idx` is the index into tp_voxels vector (usually same as panel_idx,
but for panel 5 in compare mode we use index 1 since it's an axial view).
"""
function _load_tp_into_panel!(stateObjects, tp_voxels, panel_idx, tp_voxel_idx)
    if tp_voxel_idx > length(tp_voxels) || panel_idx > length(stateObjects)
        return
    end
    newDataToScroll = StructsManag.getThreeDims(tp_voxels[tp_voxel_idx])
    stateObjects[panel_idx].onScrollData.dataToScroll = newDataToScroll
    dimToScroll = stateObjects[panel_idx].onScrollData.dimensionToScroll
    if !isempty(newDataToScroll)
        stateObjects[panel_idx].onScrollData.slicesNumber = Int32(size(newDataToScroll[1].dat, dimToScroll))
    end
    stateObjects[panel_idx].currentDisplayedSlice = max(1, stateObjects[panel_idx].onScrollData.slicesNumber ÷ 2)
end

function reactToChangeTimePoint(data::ChangeTimePointEvent, stateObjects::Vector{StateDataFields})
    if isempty(tp_data_cache)
        @info "No TP data loaded in tp_data_cache. TP navigation disabled."
        return
    end
    
    # Get sorted TP indices
    tp_indices = sort(collect(keys(tp_data_cache)))
    num_tps = length(tp_indices)
    
    # Find current position in the sorted list
    cur_pos = findfirst(==(current_tp_index[]), tp_indices)
    if cur_pos === nothing
        cur_pos = 1  # fallback to first TP
    end
    
    # Calculate new position with wrapping
    new_pos = mod1(cur_pos + data.change, num_tps)
    new_tp = tp_indices[new_pos]
    current_tp_index[] = new_tp
    
    label = get(tp_labels, new_tp, "TP $new_tp")
    @info "TP Navigation: switching to $label (index=$new_tp)"
    
    if compare_mode[]
        # Compare mode: load current TP into left panel (1), next TP into right panel (5)
        tp_voxels_left = tp_data_cache[new_tp]
        _load_tp_into_panel!(stateObjects, tp_voxels_left, 1, 1)
        
        # Right panel: next TP chronologically
        next_pos = mod1(new_pos + 1, num_tps)
        right_tp = tp_indices[next_pos]
        compare_right_tp[] = right_tp
        tp_voxels_right = tp_data_cache[right_tp]
        # Panel 5 is axial, so use voxel index 1 (the axial data)
        _load_tp_into_panel!(stateObjects, tp_voxels_right, 5, 1)
        
        # Re-render both panels
        old_idx = stateObjects[1].switchIndex
        stateObjects[1].switchIndex = 1
        ReactToScroll.reactToScroll(0, stateObjects, false)
        stateObjects[1].switchIndex = 5
        ReactToScroll.reactToScroll(0, stateObjects, false)
        stateObjects[1].switchIndex = old_idx
        
        right_label = get(tp_labels, right_tp, "TP $right_tp")
        @info "Compare: Left=$label, Right=$right_label"
    else
        # Normal mode: load current TP into all 4 panels
        tp_voxels = tp_data_cache[new_tp]
        num_panels = min(length(stateObjects), length(tp_voxels))
        for i in 1:num_panels
            _load_tp_into_panel!(stateObjects, tp_voxels, i, i)
        end
        
        # Re-render all panels
        old_idx = stateObjects[1].switchIndex
        for idx in 1:num_panels
            stateObjects[1].switchIndex = idx
            ReactToScroll.reactToScroll(0, stateObjects, false)
        end
        stateObjects[1].switchIndex = old_idx
    end
end

function reactToToggleLesion(data::ToggleLesionEvent, stateObjects::Vector{StateDataFields})
    for stateObject in stateObjects
        for textSpec in stateObject.mainForDisplayObjects.listOfTextSpecifications
            if textSpec.isMultiDiscreteMask
                textSpec.isVisible = !textSpec.isVisible
                
                # We must also push this uniform update to the GPU immediately!
                ModernGL.glUseProgram(stateObject.mainForDisplayObjects.shader_program)
                Uniforms.setTextureVisibility(textSpec.isVisible, textSpec.uniforms)
                # DO NOT break, because there could be multiple windows needing update
            end
        end
    end
end



function reactToRefreshList(data::RefreshListEvent, stateObjects::Vector{StateDataFields})
    @info "Refreshing lesion list..."
end

function reactToAddAutoPet(data::AddAutoPetEvent, stateObjects::Vector{StateDataFields})
    @info "Add New Lesion (Auto-PET) triggered."
    tp1_state = stateObjects[1]
    
    # We need CT and PET volumes. 
    # Let's assume textureSpec 1 is CT, 2 is PET, 3 is Seg. 
    # Or just search by isMainImage for CT, and PET is usually the second.
    
    ct_vol = tp1_state.mainForDisplayObjects.listOfTextSpecifications[1].imageTexture
    pet_vol = tp1_state.mainForDisplayObjects.listOfTextSpecifications[2].imageTexture
    seg_vol = tp1_state.mainForDisplayObjects.listOfTextSpecifications[3].imageTexture
    
    pos = tp1_state.lastRecordedMousePosition
    if data.algorithm == "NNInteractive"
        @info "Running NNInteractive on seed $pos"
        mask = InferenceClient.run_nninteractive(ct_vol, pos[1], pos[2], pos[3])
    else
        @info "Running HelpNet on seed $pos"
        mask = InferenceClient.run_helpnet_inference(ct_vol, pet_vol, pos[1], pos[2], pos[3])
    end
    
    if mask !== nothing
        InferenceClient.insert_patch!(seg_vol, mask, pos[1], pos[2], pos[3])
        @info "HelpNet segmented successfully."
        # Update texture uniform version or trigger render update
        # Just incrementing a dummy property to trigger render is usually how ReactToScroll works.
    else
        @warn "HelpNet inference failed or returned nothing."
    end
end

function reactToSyncMissing(data::SyncMissingEvent, stateObjects::Vector{StateDataFields})
    @info "Sync Missing Lesions across TPs triggered."
    if length(stateObjects) < 2
        @warn "Need at least 2 time points to sync missing lesions."
        return
    end
    
    tp1_state = stateObjects[1]
    tp2_state = stateObjects[2]
    
    # For now, just sync the current position. A full sync would iterate over all unique values in tp1_seg.
    pos = tp1_state.lastRecordedMousePosition
    
    tp2_ct = tp2_state.mainForDisplayObjects.listOfTextSpecifications[1].imageTexture
    tp2_pet = tp2_state.mainForDisplayObjects.listOfTextSpecifications[2].imageTexture
    tp2_seg = tp2_state.mainForDisplayObjects.listOfTextSpecifications[3].imageTexture
    
    @info "Running HelpNet on TP2 for missing lesion at $pos..."
    mask = InferenceClient.run_helpnet_inference(tp2_ct, tp2_pet, pos[1], pos[2], pos[3])
    if mask !== nothing
        InferenceClient.insert_patch!(tp2_seg, mask, pos[1], pos[2], pos[3])
        LesionAssociation.map_link("TP1", "TP2", "SyncedLesion")
        @info "Successfully synced and mapped lesion to TP2."
    end
end

function reactToGenManual(data::GenManualEvent, stateObjects::Vector{StateDataFields})
    @info "Gen Manual Lesions triggered."
    tp1_state = stateObjects[1]
    pos = tp1_state.lastRecordedMousePosition
    seg_vol = tp1_state.mainForDisplayObjects.listOfTextSpecifications[3].imageTexture
    
    cx, cy, cz = pos[1], pos[2], pos[3]
    w, h, d = size(seg_vol)
    x1, x2 = max(1, cx-2), min(w, cx+2)
    y1, y2 = max(1, cy-2), min(h, cy+2)
    z1, z2 = max(1, cz-2), min(d, cz+2)
    
    seg_vol[x1:x2, y1:y2, z1:z2] .= 1
    @info "Manual lesion generated at $pos."
end

function reactToMapLink(data::MapLinkEvent, stateObjects::Vector{StateDataFields})
    @info "Map Link triggered. Linking lesions..."
    # Normally we would link the currently active lesion from TP1 to TP2
    # This acts as a manual override matching.
    if length(stateObjects) > 1
        LesionAssociation.map_link("TP1", "TP2", data.lesion_id)
        @info "Lesion $(data.lesion_id) successfully mapped between TP1 and TP2"
    end
end

function reactToAutoRunPreprocess(data::AutoRunPreprocessEvent, stateObjects::Vector{StateDataFields})
    @info "Auto-run preprocessing toggled to $(data.active)"
end

function reactToRunPreprocess(data::RunPreprocessEvent, stateObjects::Vector{StateDataFields})
    @info "Full Preprocessing triggered."
end

function reactToShowBoneMask(data::ShowBoneMaskEvent, stateObjects::Vector{StateDataFields})
    @info "Show Bone Mask toggled to $(data.active)"
    for stateObject in stateObjects
        for textSpec in stateObject.mainForDisplayObjects.listOfTextSpecifications
            if textSpec.name == "Bone_Mask" || textSpec.name == "bone_mask" || textSpec.name == "bone" || textSpec.name == "Organ_Mask" || textSpec.name == "organ_mask"
                textSpec.isVisible = data.active
            end
        end
    end
end

function reactToSaveMRB(data::SaveMRBEvent, stateObjects::Vector{StateDataFields})
    @info "Save MRB triggered."
end

end
