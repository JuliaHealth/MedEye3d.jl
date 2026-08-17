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

function find_lesion_center(dat::AbstractArray{T, 3}, lesion_id::Float32) where T
    sum_x = 0; sum_y = 0; sum_z = 0; count = 0
    nx, ny, nz = size(dat)
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        if isapprox(dat[i, j, k], lesion_id, atol=0.1f0)
            sum_x += i
            sum_y += j
            sum_z += k
            count += 1
        end
    end
    count == 0 && return nothing
    return [round(Int, sum_x / count), round(Int, sum_y / count), round(Int, sum_z / count)]
end

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
        
        # If in compare mode, restore layout immediately
        if compare_mode[]
            if idx == 1
                updateQuadVertices!(stateObject, :LeftHalf)
            elseif idx == 5
                updateQuadVertices!(stateObject, :RightHalf)
            elseif idx in (2, 3, 4)
                updateQuadVertices!(stateObject, :Hidden)
            end
        end
        
        stateObjects[1].switchIndex = idx
        ReactToScroll.reactToScroll(0, stateObjects, false)
    end
    stateObjects[1].switchIndex = old_idx
end

function updateQuadVertices!(stateObject::StateDataFields, layout::Symbol)
    calcDimStruct = stateObject.calcDimsStruct
    
    if layout == :Hidden
        res = zeros(Float32, length(calcDimStruct.mainImageQuadVert))
        stateObject.calcDimsStruct = Setfield.setproperties(calcDimStruct, (mainImageQuadVert = res,))
    else
        pos = if layout == :TopLeft || layout == :LeftHalf
            1
        elseif layout == :TopRight || layout == :RightHalf
            2
        elseif layout == :BottomLeft
            3
        elseif layout == :BottomRight
            4
        else
            1
        end
        mode = (layout == :LeftHalf || layout == :RightHalf) ? MultiImage : QuadImage
        stateObject.calcDimsStruct = StructsManag.getMainVerticies(calcDimStruct, mode, pos)
    end
    
    ModernGL.glBindBuffer(ModernGL.GL_ARRAY_BUFFER, stateObject.mainForDisplayObjects.vbo)
    ModernGL.glBufferData(ModernGL.GL_ARRAY_BUFFER, sizeof(stateObject.calcDimsStruct.mainImageQuadVert), stateObject.calcDimsStruct.mainImageQuadVert, ModernGL.GL_STATIC_DRAW)
end

const compare_mode = Ref(false)
const compare_right_tp = Ref(-1)  # TP index shown in right panel (panel 5)

function reactToCompareTimePoints(data::CompareTimePointsEvent, stateObjects::Vector{StateDataFields})
    if length(stateObjects) >= 5
        compare_mode[] = data.compare
        if data.compare
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
                
                if length(tp_voxels) >= 5
                    _load_tp_into_panel!(stateObjects, tp_voxels, 5, 5)
                elseif length(tp_voxels) >= 1
                    # Panel 5 uses same view as panel 1 (axial), so use index 1
                    _load_tp_into_panel!(stateObjects, tp_voxels, 5, 1)
                end
            end

            # 2-pane view: panel 1 on left, panel 5 on right
            updateQuadVertices!(stateObjects[1], :LeftHalf)
            updateQuadVertices!(stateObjects[5], :RightHalf)
            updateQuadVertices!(stateObjects[2], :Hidden)
            updateQuadVertices!(stateObjects[3], :Hidden)
            updateQuadVertices!(stateObjects[4], :Hidden)

            # Re-render both panels by clearing current display data to force texture update
            stateObjects[1].currentlyDispDat = SingleSliceDat()
            stateObjects[5].currentlyDispDat = SingleSliceDat()
            old_idx = stateObjects[1].switchIndex
            stateObjects[1].switchIndex = 1
            ReactToScroll.reactToScroll(0, stateObjects, false)
            stateObjects[1].switchIndex = 5
            ReactToScroll.reactToScroll(0, stateObjects, false)
            stateObjects[1].switchIndex = old_idx
            
            if !isempty(tp_data_cache)
                left_label = get(tp_labels, current_tp_index[], "TP $(current_tp_index[])")
                right_label = get(tp_labels, right_tp, "TP $right_tp")
                @info "Compare mode ON: Left=$left_label, Right=$right_label"
            end
        else
            compare_right_tp[] = -1
            # Reload current active TP into all 4 panels so all views (Axial, PET, Sagittal, Coronal) match current_tp_index[]
            if !isempty(tp_data_cache) && haskey(tp_data_cache, current_tp_index[])
                tp_voxels = tp_data_cache[current_tp_index[]]
                num_panels = min(4, length(stateObjects))
                for i in 1:num_panels
                    _load_tp_into_panel!(stateObjects, tp_voxels, i)
                end
            end

            # 4-pane view
            updateQuadVertices!(stateObjects[1], :TopLeft)
            updateQuadVertices!(stateObjects[2], :TopRight)
            updateQuadVertices!(stateObjects[3], :BottomLeft)
            updateQuadVertices!(stateObjects[4], :BottomRight)
            updateQuadVertices!(stateObjects[5], :Hidden)
            
            # Reset pan, zoom, displayMode, and center slice for all 4 panes
            for i in 1:4
                stateObjects[i].calcDimsStruct.zoom = 1.0f0
                stateObjects[i].calcDimsStruct.panX = 0.0f0
                stateObjects[i].calcDimsStruct.panY = 0.0f0
                stateObjects[i].displayMode = QuadImage
                if stateObjects[i].onScrollData.slicesNumber > 0
                    stateObjects[i].currentDisplayedSlice = max(1, stateObjects[i].onScrollData.slicesNumber ÷ 2)
                end
            end

            # Re-render all 4 panels to ensure textures and slices are displayed
            old_idx = stateObjects[1].switchIndex
            for i in 1:4
                stateObjects[i].currentlyDispDat = SingleSliceDat()
                stateObjects[1].switchIndex = i
                ReactToScroll.reactToScroll(0, stateObjects, false)
            end
            stateObjects[1].switchIndex = old_idx
            @info "Compare mode OFF: restored 4-pane view for TP $(current_tp_index[])"
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
            if tex.name == "CT"
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
        state.valueForMasToSet = valueForMasToSetStruct(value=data.val, is_painting_active=data.active)
    end
    @info "Paint state updated: val=$(data.val), active=$(data.active)"
end

const tp_node_names = Dict{Int, String}()

function get_node_name_for_tp(tp_idx::Int)::String
    if haskey(tp_node_names, tp_idx)
        return tp_node_names[tp_idx]
    end
    lbl = get(tp_labels, tp_idx, "")
    if occursin("PET", lbl) && occursin("TP", lbl)
        m = match(r"TP\s*(\d+)", lbl)
        if m !== nothing
            return "PET_Lesions_$(m.captures[1])"
        end
    elseif occursin("SPECT", lbl) && occursin("TP", lbl)
        m = match(r"TP\s*(\d+)", lbl)
        if m !== nothing
            return "SPECT_Lesions_$(m.captures[1])"
        end
    end
    return "PET_Lesions_$tp_idx"
end

function reactToSyncLesion(data::SyncLesionEvent, stateObjects::Vector{StateDataFields})
    @info "reactToSyncLesion called with lesion_id=$(data.lesion_id), nStates=$(length(stateObjects))"
    changed = false
    old_idx = stateObjects[1].switchIndex
    old_sync = stateObjects[1].mainForDisplayObjects.isSyncScrollOn
    for stateObject in stateObjects
        stateObject.mainForDisplayObjects.isSyncScrollOn = false
    end

    # Determine matched lesion ID for the compare right panel (panel 5) if in compare mode
    panel5_lesion_id = data.lesion_id
    if compare_mode[] && length(stateObjects) >= 5 && data.lesion_id > 0
        try
            left_node = get_node_name_for_tp(current_tp_index[])
            right_node = get_node_name_for_tp(compare_right_tp[])
            # Find cross-TP match from LesionAssociation module
            match_mod = isdefined(Main, :MedEye3d) && isdefined(Main.MedEye3d, :LesionAssociation) ? 
                        Main.MedEye3d.LesionAssociation : nothing
            if match_mod !== nothing
                matched_ids = match_mod.find_cross_tp_lesion(left_node, data.lesion_id, right_node)
                if !isempty(matched_ids)
                    panel5_lesion_id = matched_ids[1]
                    @info "Cross-TP match: $(left_node) lesion $(data.lesion_id) -> $(right_node) lesion $(panel5_lesion_id)"
                end
            end
        catch e
            @warn "Error finding cross-TP lesion: $e"
        end
    end

    # Set mask filter uniform for each panel
    for (idx, stateObject) in enumerate(stateObjects)
        target_id = (idx == 5 && compare_mode[]) ? panel5_lesion_id : data.lesion_id
        for textSpec in stateObject.mainForDisplayObjects.listOfTextSpecifications
            if textSpec.isMultiDiscreteMask || textSpec.name == "Mask"
                if target_id > 0
                    textSpec.minAndMaxValue = Float32.([target_id, target_id])
                else
                    textSpec.minAndMaxValue = Float32.([1.0, 1000.0])
                end
                ModernGL.glUseProgram(stateObject.mainForDisplayObjects.shader_program)
                Uniforms.coontrolMinMaxUniformVals(textSpec)
                @info "Set mask uniform for panel $idx: lesion=$target_id, texSpec.name=$(textSpec.name)"
            end
        end
    end

    active_panel_indices = if compare_mode[] && length(stateObjects) >= 5
        [1, 5]
    elseif length(stateObjects) >= 4
        [1, 2, 3, 4]
    else
        collect(1:length(stateObjects))
    end

    canonical_center = nothing
    if data.lesion_id > 0
        # Find lesion center in axial panel 1 (or any panel with Mask)
        for (si, stateObject) in enumerate(stateObjects)
            for (scrIdx, scrDat) in enumerate(stateObject.onScrollData.dataToScroll)
                texSpec = stateObject.mainForDisplayObjects.listOfTextSpecifications[scrIdx]
                if (texSpec.name == "Mask" || texSpec.name == "manualModif") && stateObject.onScrollData.dimensionToScroll == 3
                    canonical_center = find_lesion_center(scrDat.dat, Float32(data.lesion_id))
                    if canonical_center !== nothing
                        break
                    end
                end
            end
            canonical_center !== nothing && break
        end
    end
    @info "canonical_center=$canonical_center, active_panel_indices=$active_panel_indices"

    if canonical_center !== nothing
        for idx in active_panel_indices
            stateObject = stateObjects[idx]
            last_sl = max(1, stateObject.onScrollData.slicesNumber)
            
            # Check if this panel has its own lesion center (especially in compare mode)
            target_id = (idx == 5 && compare_mode[]) ? panel5_lesion_id : data.lesion_id
            panel_center = nothing
            if idx == 5 && compare_mode[] && target_id > 0
                for (scrIdx, scrDat) in enumerate(stateObject.onScrollData.dataToScroll)
                    texSpec = stateObject.mainForDisplayObjects.listOfTextSpecifications[scrIdx]
                    if (texSpec.name == "Mask" || texSpec.name == "manualModif") && stateObject.onScrollData.dimensionToScroll == 3
                        panel_center = find_lesion_center(scrDat.dat, Float32(target_id))
                        panel_center !== nothing && break
                    end
                end
            end
            
            effective_center = panel_center !== nothing ? panel_center : canonical_center
            origX, origY, origZ = effective_center[1], effective_center[2], effective_center[3]
            
            if idx == 1 || idx == 2 || idx == 5
                # Axial view (scrolls Z, shows X vs Y)
                stateObject.lastRecordedMousePosition = CartesianIndex(origX, origY, origZ)
                stateObject.currentDisplayedSlice = clamp(origZ, 1, last_sl)
                texX, texY = origX, origY
            elseif idx == 3
                # Sagittal view (permuted 2,3,1: Y, Z, X; scrolls X, shows Y vs Z)
                stateObject.lastRecordedMousePosition = CartesianIndex(origY, origZ, origX)
                stateObject.currentDisplayedSlice = clamp(origX, 1, last_sl)
                texX, texY = origY, origZ
            else
                # Coronal view (idx 4, permuted 1,3,2: X, Z, Y; scrolls Y, shows X vs Z)
                stateObject.lastRecordedMousePosition = CartesianIndex(origX, origZ, origY)
                stateObject.currentDisplayedSlice = clamp(origY, 1, last_sl)
                texX, texY = origX, origZ
            end
            
            # Center the view on the lesion if zoomed
            if !isempty(stateObject.onScrollData.dataToScroll)
                dat_shape = size(stateObject.onScrollData.dataToScroll[1].dat)
                h_img = dat_shape[1]
                w_img = dat_shape[2]
                stateObject.calcDimsStruct.panX = Float32((texY - w_img / 2) / w_img)
                stateObject.calcDimsStruct.panY = Float32((texX - h_img / 2) / h_img)
            end
            
            stateObjects[1].switchIndex = idx
            ReactToScroll.reactToScroll(0, stateObjects, false)
            changed = true
            @info "Synced active lesion $target_id at center $effective_center in panel $idx"
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
const tp_descriptions = Dict{Int, String}() # tp_index → radiological description
export tp_data_cache, current_tp_index, tp_labels, tp_descriptions
export compare_mode, compare_right_tp

"""
Helper: load TP data into a specific panel's onScrollData and re-render.
`panel_idx` is the stateObjects index (1-5).
`tp_voxel_idx` is the index into tp_voxels vector (usually same as panel_idx,
but for panel 5 in compare mode we use index 1 since it's an axial view).
"""
function _load_tp_into_panel!(stateObjects, tp_voxels, panel_idx, tp_voxel_idx=panel_idx)
    if panel_idx > length(stateObjects) || tp_voxel_idx > length(tp_voxels)
        return
    end
    
    panel_voxels = tp_voxels[tp_voxel_idx]
    
    # Ensure manualModif is inserted at index 2 if it's missing (to match SegmentationDisplay.jl initialization)
    if length(panel_voxels) >= 1 && panel_voxels[1][1] != "manualModif" && (length(panel_voxels) < 2 || panel_voxels[2][1] != "manualModif")
        insert!(panel_voxels, 2, ("manualModif", zeros(Float32, size(panel_voxels[1][2]))))
    end
    
    newDataToScroll = StructsManag.getThreeDims(panel_voxels)
    stateObjects[panel_idx].onScrollData.dataToScroll = newDataToScroll
    stateObjects[panel_idx].onScrollData.nameIndexes = DataStructs.getLocationDict(newDataToScroll)
    
    dimToScroll = stateObjects[panel_idx].onScrollData.dimensionToScroll
    if !isempty(newDataToScroll)
        stateObjects[panel_idx].onScrollData.slicesNumber = Int32(size(newDataToScroll[1].dat, dimToScroll))
    end
    stateObjects[panel_idx].currentDisplayedSlice = max(1, stateObjects[panel_idx].onScrollData.slicesNumber ÷ 2)
    stateObjects[panel_idx].currentlyDispDat = SingleSliceDat()
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
    
    # Requirement: Automatically return to Lesion 1 when changing time points
    try
        reactToSyncLesion(SyncLesionEvent(1), stateObjects)
        @info "Auto-reset to Lesion 1 for $label"
    catch e
        @warn "Failed to auto-sync Lesion 1 on TP change: $e"
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
    @info "Bone subsegmentation triggered for lesion $(data.lesion_id)"
    tp1_state = stateObjects[1]
    
    seg_vol = tp1_state.mainForDisplayObjects.listOfTextSpecifications[3].imageTexture
    ct_vol = tp1_state.mainForDisplayObjects.listOfTextSpecifications[2].imageTexture
    
    bone_atlas = Float32.(ct_vol .> 150.0f0)
    spacing = (1.5f0, 1.5f0, 2.0f0)
    
    try
        surf, marr = MedEye3d.BoneSubsegmentation.generate_bone_subsegments(seg_vol, bone_atlas, spacing, data.lesion_id)
        seg_vol[surf] .= 2.0f0
        seg_vol[marr] .= 3.0f0
        @info "Bone subsegments generated successfully."
        reactToScroll(0, stateObjects, false)
    catch e
        @error "Failed to generate bone subsegments: $e"
    end
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

const MASK_BACKUP = Dict{UInt64, Array{Float32, 3}}()

function reactToShowMaskLayer(data::ShowMaskLayerEvent, stateObjects::Vector{StateDataFields})
    @info "ShowMaskLayerEvent: layer $(data.layer) -> $(data.active)"
    
    for state in stateObjects
        for (i, scrDat) in enumerate(state.onScrollData.dataToScroll)
            texSpec = state.mainForDisplayObjects.listOfTextSpecifications[i]
            if texSpec.name == "Mask" || texSpec.name == "manualModif"
                dat_ptr = pointer(scrDat.dat)
                dat_key = UInt64(UInt(dat_ptr))
                
                if !haskey(MASK_BACKUP, dat_key)
                    MASK_BACKUP[dat_key] = copy(scrDat.dat)
                end
                
                backup_vol = MASK_BACKUP[dat_key]
                target_val = Float32(data.layer)
                
                if data.active
                    idx = findall(backup_vol .== target_val)
                    scrDat.dat[idx] .= target_val
                else
                    idx = findall(scrDat.dat .== target_val)
                    scrDat.dat[idx] .= 0.0f0
                end
            end
        end
        
        # Update single slice data for this pane from the modified volume
        singleSlDat = state.onScrollData.dataToScroll |>
            (scrDat) -> map(threeDimDat -> threeToTwoDimm(threeDimDat.type, Int64(state.currentDisplayedSlice), state.onScrollData.dimensionToScroll, threeDimDat), scrDat) |>
            (twoDimList) -> SingleSliceDat(listOfDataAndImageNames=twoDimList, sliceNumber=state.currentDisplayedSlice, textToDisp=getTextForCurrentSlice(state.onScrollData, Int32(state.currentDisplayedSlice)))
            
        state.currentlyDispDat = singleSlDat
        
        # Upload new texture data to GPU
        for updateDat in singleSlDat.listOfDataAndImageNames
            findList = findall((texSpec) -> texSpec.name == updateDat.name, state.mainForDisplayObjects.listOfTextSpecifications)
            if !isempty(findList)
                texSpec = state.mainForDisplayObjects.listOfTextSpecifications[findList[1]]
                transformedDat = applyZoomPan(updateDat.dat, state.calcDimsStruct.zoom, state.calcDimsStruct.panX, state.calcDimsStruct.panY)
                updateTexture(updateDat.type, transformedDat, texSpec, 0, 0, state.calcDimsStruct.imageTextureWidth, state.calcDimsStruct.imageTextureHeight)
            end
        end
    end
    
    # Re-render all panels
    old_idx = stateObjects[1].switchIndex
    for idx in 1:length(stateObjects)
        stateObjects[1].switchIndex = idx
        ReactToScroll.reactToScroll(0, stateObjects, false)
    end
    stateObjects[1].switchIndex = old_idx
end

function reactToSaveMRB(data::SaveMRBEvent, stateObjects::Vector{StateDataFields})
    @info "Save MRB triggered."
end

end
