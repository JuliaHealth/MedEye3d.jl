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

function reactToCompareTimePoints(data::CompareTimePointsEvent, stateObjects::Vector{StateDataFields})
    if length(stateObjects) >= 2
        for textSpec in stateObjects[2].mainForDisplayObjects.listOfTextSpecifications
            textSpec.isVisible = data.compare
            
            ModernGL.glUseProgram(stateObjects[2].mainForDisplayObjects.shader_program)
            Uniforms.setTextureVisibility(textSpec.isVisible, textSpec.uniforms)
        end
        # The render loop will use the updated uniforms in `activateForMainDisp` ?
        # Actually `activateForMainDisp` loops over `listOfTextSpecifications` and calls `setSingleTextureVisib(textSpec.uniforms, textSpec.isVisible)`
    end
end

function reactToShowSingleLesion(data::ShowSingleLesionEvent, stateObjects::Vector{StateDataFields})
    old_idx = stateObjects[1].switchIndex
    for (idx, stateObject) in enumerate(stateObjects)
        if data.lesion_id > 0
            for (scrIdx, scrDat) in enumerate(stateObject.onScrollData.dataToScroll)
                texSpec = stateObject.mainForDisplayObjects.listOfTextSpecifications[scrIdx]
                if texSpec.name == "Mask" || texSpec.name == "manualModif"
                    indices = findall(x -> isapprox(x, Float32(data.lesion_id), atol=0.1f0), scrDat.dat)
                    if !isempty(indices)
                        center = round.(Int, Tuple(sum(indices)) ./ length(indices))
                        stateObject.lastRecordedMousePosition = CartesianIndex(center[1], center[2], center[3])
                        
                        dim = stateObject.onScrollData.dimensionToScroll
                        stateObject.currentDisplayedSlice = center[dim]
                        
                        stateObjects[1].switchIndex = idx
                        ReactToScroll.reactToScroll(0, stateObjects, false)
                        @info "Navigated to lesion $(data.lesion_id) at center $center in window $idx"
                        break
                    end
                end
            end
        end
        
        for textSpec in stateObject.mainForDisplayObjects.listOfTextSpecifications
            if !textSpec.isMainImage && !textSpec.isContinuusMask
                if data.lesion_id == 0
                    textSpec.minAndMaxValue = Float32.([1.0, 1000.0])
                else
                    textSpec.minAndMaxValue = Float32.([data.lesion_id, data.lesion_id])
                end
                
                # Push uniform update for min/max
                ModernGL.glUseProgram(stateObject.mainForDisplayObjects.shader_program)
                Uniforms.coontrolMinMaxUniformVals(textSpec)
            end
        end
    end
    stateObjects[1].switchIndex = old_idx
end

function reactToWindowing(data::WindowingEvent, stateObjects::Vector{StateDataFields})
    for state in stateObjects
        for tex in state.mainForDisplayObjects.listOfTextSpecifications
            if !tex.isNuclearMask && !tex.isContinuusMask
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
    old_idx = stateObjects[1].switchIndex
    old_sync = stateObjects[1].mainForDisplayObjects.isSyncScrollOn
    for stateObject in stateObjects
        stateObject.mainForDisplayObjects.isSyncScrollOn = false
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
                        found_center = round.(Int, Tuple(sum(indices)) ./ length(indices))
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
                    stateObject.lastRecordedMousePosition = mapped_pos
                    dim = stateObject.onScrollData.dimensionToScroll
                    stateObject.currentDisplayedSlice = mapped_pos[dim]
                    
                    stateObjects[1].switchIndex = idx
                    ReactToScroll.reactToScroll(0, stateObjects, false)
                    @info "Mapped lesion $(data.lesion_id) via physical coordinates to window $idx: $mapped_pos"
                end
            end
        end
    end
    for stateObject in stateObjects
        stateObject.mainForDisplayObjects.isSyncScrollOn = old_sync
    end
    stateObjects[1].switchIndex = old_idx
end

function reactToChangeTimePoint(data::ChangeTimePointEvent, stateObjects::Vector{StateDataFields})
    @info "TP Navigation: $(data.change). Emulated."
end

function reactToToggleLesion(data::ToggleLesionEvent, stateObjects::Vector{StateDataFields})
    for stateObject in stateObjects
        for textSpec in stateObject.mainForDisplayObjects.listOfTextSpecifications
            if !textSpec.isMainImage && !textSpec.isContinuusMask
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
