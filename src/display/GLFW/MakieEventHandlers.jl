module MakieEventHandlers
using ...MakieEvents
using ...StructsManag
using ...ForDisplayStructs
using ...DataStructs
using ...ChangePlane
using ...ReactToScroll
using DataTypesBasic
using Setfield
using Statistics: mean
using Dates

export reactToChangePlane, reactToCompareTimePoints, reactToShowSingleLesion
export reactToWindowing, reactToPaintVal, reactToSyncLesion, reactToChangeBrushSize, reactToPetBlend
export reactToChangeTimePoint, reactToToggleLesion, reactToRefreshList
export reactToAddAutoPet, reactToAIInferenceResult, reactToSyncMissing, reactToGenManual
export reactToMapLink, reactToAutoRunPreprocess, reactToRunPreprocess, reactToShowBoneMask, reactToShowMaskLayer, reactToSaveMRB
using ...InferenceClient
using ...LesionAssociation
using ...TextureManag
using ModernGL
using ..Uniforms
using Observables

# AI status Observable — LesionMetadataWindow reads this for the GUI label
const ai_status_text = Observable{String}("Ready")

# Sanitize AI status text for Makie Label rendering (ASCII-only, truncated)
function safe_status_text(msg::String)
    s = replace(msg, "\u2014" => "-", "\u2026" => "...")
    s = String(filter(c -> isascii(c), collect(s)))
    return length(s) > 80 ? s[1:80] * "..." : s
end

# Internal inference queue — serializes all Docker communication through a single worker thread
struct InferenceJob
    algorithm::String
    ct_vol::Array{Float32, 3}
    pet_vol::Array{Float32, 3}
    points_vol::Array{Float32, 3}
    cx::Int
    cy::Int
    cz::Int
    active_id::Int
    seg_vol::Union{Nothing, Array{Float32, 3}}
    main_channel::Channel{Any}
    scribble_coords::Vector{Vector{Int}}  # Pre-extracted 0-indexed [x,y,z] coords for nnInteractive fast path
end

const inference_queue = Channel{InferenceJob}(8)

# Persistent worker thread — started once, processes jobs sequentially (no race conditions)
function start_inference_worker()
    Threads.@spawn begin
        println("[AI Worker] Inference worker thread started."); flush(stdout)
        while true
            try
                job = take!(inference_queue)
                
                ai_status_text[] = safe_status_text("[Sending] to Docker ($(job.algorithm))...")
                println("[AI Worker] Processing $(job.algorithm) at ($(job.cx),$(job.cy),$(job.cz)) for lesion $(job.active_id)..."); flush(stdout)
                
                mask = nothing
                if job.algorithm == "NNInteractive"
                    # Fast path: use pre-extracted scribble coordinates (skip findall)
                    if !isempty(job.scribble_coords)
                        mask = InferenceClient.run_nninteractive(
                            job.ct_vol, job.pet_vol, job.scribble_coords,
                            job.cx, job.cy, job.cz)
                    else
                        mask = InferenceClient.run_nninteractive(
                            job.ct_vol, job.pet_vol, job.points_vol,
                            job.cx, job.cy, job.cz)
                    end
                elseif job.algorithm == "HELPNet (AI)"
                    mask = InferenceClient.run_helpnet_inference(
                        job.ct_vol, job.pet_vol, job.points_vol,
                        job.cx, job.cy, job.cz)
                else
                    println("[AI Worker] WARNING: Unknown algorithm: $(job.algorithm)"); flush(stdout)
                    ai_status_text[] = safe_status_text("[Warning] Unknown algorithm: $(job.algorithm)")
                    continue
                end
                
                if mask !== nothing
                    voxel_count = count(mask .> 0)
                    ai_status_text[] = safe_status_text("[Applying] result ($voxel_count voxels)...")
                    println("[AI Worker] Docker returned mask with $voxel_count voxels. Posting to channel."); flush(stdout)
                else
                    ai_status_text[] = safe_status_text("[Warning] Docker returned no mask")
                    println("[AI Worker] Docker returned nothing (inference failed)."); flush(stdout)
                end
                
                # Post result back to main event channel via on_next! multiple dispatch
                put!(job.main_channel, AIInferenceResultEvent(
                    job.algorithm, job.active_id,
                    job.cx, job.cy, job.cz,
                    mask, job.seg_vol))
                    
            catch e
                if e isa InvalidStateException  # channel closed
                    println("[AI Worker] Queue closed, shutting down."); flush(stdout)
                    break
                end
                err_msg = sprint(showerror, e)
                println("[AI Worker] ERROR: $err_msg"); flush(stdout)
                println(sprint(showerror, e, catch_backtrace())); flush(stdout)
                ai_status_text[] = safe_status_text("[Error] AI Worker Error: $err_msg")
                try
                    open("/tmp/medeye3d_errors.log", "a") do f
                        println(f, "$(Dates.now()) AI Worker ERROR: $err_msg")
                        println(f, sprint(showerror, e, catch_backtrace()))
                        println(f, "---")
                    end
                catch; end
            end
        end
    end
end

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
                println("Compare mode ON: Left=$left_label, Right=$right_label"); flush(stdout)
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
            println("Compare mode OFF: restored 4-pane view for TP $(current_tp_index[])"); flush(stdout)
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
    lbl = data.lesion_id == 0 ? "all" : string(data.lesion_id)
    println("Show single lesion: $lbl"); flush(stdout)
    return changed
end

const current_windowing = Dict{String, Vector{Float32}}(
    "CT" => Float32[-150.0, 250.0],
    "PET" => Float32[0.0, 10.0],
    "SPECT" => Float32[0.0, 10.0]
)
export current_windowing

function reactToWindowing(data::WindowingEvent, stateObjects::Vector{StateDataFields})
    target_mod = uppercase(data.modality)
    current_windowing[target_mod] = Float32.([data.min_val, data.max_val])
    
    for state in stateObjects
        for tex in state.mainForDisplayObjects.listOfTextSpecifications
            match = (target_mod == "CT" && tex.name == "CT") || 
                    ((target_mod == "PET" || target_mod == "SPECT") && tex.name == "PET")
            if match
                tex.minAndMaxValue = Float32.([data.min_val, data.max_val])
                
                # Push uniform update for min/max
                ModernGL.glUseProgram(state.mainForDisplayObjects.shader_program)
                Uniforms.coontrolMinMaxUniformVals(tex)
            end
        end
    end
    println("Updated windowing for $(data.modality): [$(data.min_val), $(data.max_val)]"); flush(stdout)
end

function reactToPetBlend(data::PetBlendEvent, stateObjects::Vector{StateDataFields})
    for state in stateObjects
        ModernGL.glUseProgram(state.mainForDisplayObjects.shader_program)
        for tex in state.mainForDisplayObjects.listOfTextSpecifications
            # Update PET overlay contribution (not the pure PET main image panel)
            if tex.name == "PET" && !tex.isMainImage
                tex.maskContribution = data.weight
                @uniforms! begin
                    tex.uniforms.maskContribution := data.weight
                end
            end
        end
    end
    println("PET/CT blend updated to $(data.weight)"); flush(stdout)
end

function reactToPaintVal(data::PaintValEvent, stateObjects::Vector{StateDataFields})
    for state in stateObjects
        state.valueForMasToSet = valueForMasToSetStruct(value=data.val, is_painting_active=data.active)
        if data.active
            for textSpec in state.mainForDisplayObjects.listOfTextSpecifications
                if textSpec.name == "manualModif"
                    textSpec.isVisible = true
                    ModernGL.glUseProgram(state.mainForDisplayObjects.shader_program)
                    Uniforms.setTextureVisibility(textSpec.isVisible, textSpec.uniforms)
                end
            end
        end
    end
    println("Paint state updated: val=$(data.val), active=$(data.active)"); flush(stdout)
end

function reactToChangeBrushSize(data::ChangeBrushSizeEvent, stateObjects::Vector{StateDataFields})
    for state in stateObjects
        if !isempty(state.textureToModifyVec)
            state.textureToModifyVec[1].strokeWidth = Int32(data.size)
        end
    end
    println("Brush size updated to $(data.size)"); flush(stdout)
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

const current_active_lesion_id = Ref(0)

function reactToSyncLesion(data::SyncLesionEvent, stateObjects::Vector{StateDataFields})
    println("reactToSyncLesion called with lesion_id=$(data.lesion_id), nStates=$(length(stateObjects))"); flush(stdout)
    current_active_lesion_id[] = data.lesion_id
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
                    println("Cross-TP match: $(left_node) lesion $(data.lesion_id) -> $(right_node) lesion $(panel5_lesion_id)"); flush(stdout)
                end
            end
        catch e
            println("WARNING: Error finding cross-TP lesion: $e"); flush(stdout)
        end
    end

    # Set mask filter uniform for each panel
    for (idx, stateObject) in enumerate(stateObjects)
        target_id = (idx == 5 && compare_mode[]) ? panel5_lesion_id : data.lesion_id
        for textSpec in stateObject.mainForDisplayObjects.listOfTextSpecifications
            if textSpec.name == "Mask" || textSpec.name == "segmentation"
                if target_id > 0
                    textSpec.minAndMaxValue = Float32.([target_id, target_id])
                else
                    textSpec.minAndMaxValue = Float32.([1.0, 1000.0])
                end
                ModernGL.glUseProgram(stateObject.mainForDisplayObjects.shader_program)
                Uniforms.coontrolMinMaxUniformVals(textSpec)
                println("Set mask uniform for panel $idx: lesion=$target_id, texSpec.name=$(textSpec.name)"); flush(stdout)
            elseif textSpec.name == "manualModif"
                # manualModif must always remain unclamped so all user strokes are visible
                textSpec.minAndMaxValue = Float32.([0.0, 1000.0])
                ModernGL.glUseProgram(stateObject.mainForDisplayObjects.shader_program)
                Uniforms.coontrolMinMaxUniformVals(textSpec)
            end
        end
    end

    # Sparse fast bone subsegments update (instantaneous, zero large array allocations)
    if data.lesion_id > 0
        if !haskey(bone_subsegments_cache, data.lesion_id) && global_bone_atlas[] !== nothing
            try
                seg_vol = nothing
                for dat in stateObjects[1].onScrollData.dataToScroll
                    if dat.name == "Mask" || dat.name == "segmentation"
                        seg_vol = dat.dat
                        break
                    end
                end
                if seg_vol !== nothing
                    # Check if lesion overlaps with bone atlas before computing bone subsegments
                    lesion_indices = findall(seg_vol .== Float32(data.lesion_id))
                    bone_atlas = global_bone_atlas[]
                    has_bone_overlap = !isempty(lesion_indices) && any(idx -> checkbounds(Bool, bone_atlas, idx) && bone_atlas[idx] > 0.0f0, lesion_indices)
                    if has_bone_overlap
                        spacing = (1.5f0, 1.5f0, 2.0f0)
                        surf_mask, marr_mask = MedEye3d.BoneSubsegmentation.generate_bone_subsegments(seg_vol, bone_atlas, spacing, data.lesion_id)
                        pts_surf = findall(surf_mask)
                        pts_marr = findall(marr_mask)
                        bone_subsegments_cache[data.lesion_id] = (pts_surf, pts_marr)
                        println("Generated on-the-fly bone subsegments for synced lesion $(data.lesion_id): $(length(pts_surf)) surface, $(length(pts_marr)) marrow"); flush(stdout)
                    else
                        # Non-bone lesion: cache empty arrays so we don't recompute every navigation
                        bone_subsegments_cache[data.lesion_id] = (CartesianIndex{3}[], CartesianIndex{3}[])
                        println("Lesion $(data.lesion_id) does not overlap bone atlas — skipping bone subsegmentation"); flush(stdout)
                    end
                end
            catch e
                println("WARNING: Failed to compute on-the-fly bone subsegments for lesion $(data.lesion_id): $e"); flush(stdout)
            end
        end
    end

    for (panel_idx, stateObject) in enumerate(stateObjects)
        surf_indices = CartesianIndex{3}[]
        marr_indices = CartesianIndex{3}[]
        
        if data.lesion_id > 0 && haskey(bone_subsegments_cache, data.lesion_id)
            raw_surf, raw_marr = bone_subsegments_cache[data.lesion_id]
            
            # Support both CartesianIndex[] sparse cache and raw 3D array cache
            pts_surf = raw_surf isa AbstractArray{<:CartesianIndex} ? raw_surf : findall(raw_surf .> 0)
            pts_marr = raw_marr isa AbstractArray{<:CartesianIndex} ? raw_marr : findall(raw_marr .> 0)
            
            if panel_idx == 3 # Sagittal (Y, Z, X)
                surf_indices = [CartesianIndex(I[2], I[3], I[1]) for I in pts_surf]
                marr_indices = [CartesianIndex(I[2], I[3], I[1]) for I in pts_marr]
            elseif panel_idx == 4 # Coronal (X, Z, Y)
                surf_indices = [CartesianIndex(I[1], I[3], I[2]) for I in pts_surf]
                marr_indices = [CartesianIndex(I[1], I[3], I[2]) for I in pts_marr]
            else # Axial (X, Y, Z)
                surf_indices = pts_surf
                marr_indices = pts_marr
            end
        end
        
        for scrDat in stateObject.onScrollData.dataToScroll
            if scrDat.name == "Bone_Surface"
                fill!(scrDat.dat, 0.0f0)
                if !isempty(surf_indices)
                    scrDat.dat[surf_indices] .= 1.0f0
                end
            elseif scrDat.name == "Bone_Marrow"
                fill!(scrDat.dat, 0.0f0)
                if !isempty(marr_indices)
                    scrDat.dat[marr_indices] .= 1.0f0
                end
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

    canonical_center = if data.lesion_id > 0 && haskey(lesion_centroids_cache, data.lesion_id)
        lesion_centroids_cache[data.lesion_id]
    elseif data.lesion_id > 0
        # on-the-fly centroid computation when not pre-cached
        cc = nothing
        for (si, stateObject) in enumerate(stateObjects)
            for (scrIdx, scrDat) in enumerate(stateObject.onScrollData.dataToScroll)
                texSpec = stateObject.mainForDisplayObjects.listOfTextSpecifications[scrIdx]
                if (texSpec.name == "Mask" || texSpec.name == "manualModif") && stateObject.onScrollData.dimensionToScroll == 3
                    cc = find_lesion_center(scrDat.dat, Float32(data.lesion_id))
                    cc !== nothing && break
                end
            end
            cc !== nothing && break
        end
        cc
    else
        nothing
    end
    println("canonical_center=$canonical_center, active_panel_indices=$active_panel_indices"); flush(stdout)

    if canonical_center !== nothing
        for idx in active_panel_indices
            stateObject = stateObjects[idx]
            last_sl = max(1, stateObject.onScrollData.slicesNumber)
            
            # Check if this panel has its own lesion center (especially in compare mode)
            target_id = (idx == 5 && compare_mode[]) ? panel5_lesion_id : data.lesion_id
            panel_center = if idx == 5 && compare_mode[] && target_id > 0 && haskey(lesion_centroids_cache, target_id)
                lesion_centroids_cache[target_id]
            else
                nothing
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
            
            stateObjects[1].switchIndex = idx
            ReactToScroll.reactToScroll(0, stateObjects, false)
            changed = true
            println("Synced active lesion $target_id at center $effective_center in panel $idx (slice $(stateObject.currentDisplayedSlice))"); flush(stdout)
        end
    end
    for stateObject in stateObjects
        stateObject.mainForDisplayObjects.isSyncScrollOn = old_sync
    end
    stateObjects[1].switchIndex = old_idx
    return changed
end

# TP navigation state: populated by the launch script
# tp_data_cache[tp_index] = Vector{Vector{Any}} — per-panel voxel data tuples [("CT", vol), ("PET", vol), ("Mask", vol)]
const tp_data_cache = Dict{Int, Vector{Vector{Any}}}()
const bone_subsegments_cache = Dict{Int, Any}()
const lesion_centroids_cache = Dict{Int, Vector{Int}}()
const global_bone_atlas = Ref{Any}(nothing)
const global_organ_mapping = Ref{Dict{Int,String}}(Dict{Int,String}())  # lesion_id -> TS organ name (from map_lesions_to_organs)
const current_tp_index = Ref(0)
const tp_labels = Dict{Int, String}()  # tp_index -> display label (e.g. "PET TP0")
const tp_descriptions = Dict{Int, String}() # tp_index -> radiological description

# PET volume per TP for SUV computation: tp_index -> 3D Float32 array (axial orientation, Y-reversed)
const pet_volumes_cache = Dict{Int, Array{Float32, 3}}()
# TotalSegmentator atlas + names for background SUV reference organs
const global_ts_atlas = Ref{Any}(nothing)          # 3D UInt8/Int array (axial, Y-reversed)
const global_ts_names = Ref{Dict{Int,String}}(Dict{Int,String}())  # TS label -> organ name
# Patient identification
const patient_id = Ref{String}("")
# Study modalities per TP: tp_index -> "PET" or "SPECT"
const tp_modalities = Dict{Int, String}()

export tp_data_cache, bone_subsegments_cache, lesion_centroids_cache, global_bone_atlas, global_organ_mapping, current_tp_index, tp_labels, tp_descriptions
export compare_mode, compare_right_tp
export pet_volumes_cache, global_ts_atlas, global_ts_names, patient_id, tp_modalities

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
        println("No TP data loaded in tp_data_cache. TP navigation disabled."); flush(stdout)
        return
    end
    
    # Get sorted TP indices
    tp_indices = sort(collect(keys(tp_data_cache)))
    num_tps = length(tp_indices)
    
    # Find current position in the sorted list
    cur_pos = findfirst(==(current_tp_index[]), tp_indices)
    if cur_pos === nothing
        cur_pos = 1  # default to first TP if current index not found
    end
    
    # Calculate new position with wrapping
    new_pos = mod1(cur_pos + data.change, num_tps)
    new_tp = tp_indices[new_pos]
    current_tp_index[] = new_tp
    
    label = get(tp_labels, new_tp, "TP $new_tp")
    println("TP Navigation: switching to $label (index=$new_tp)"); flush(stdout)
    
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
        println("Compare: Left=$label, Right=$right_label"); flush(stdout)
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
        println("Auto-reset to Lesion 1 for $label"); flush(stdout)
    catch e
        println("WARNING: Failed to auto-sync Lesion 1 on TP change: $e"); flush(stdout)
    end
    
    # Preload the new CT into Docker nnInteractive GPU memory (fire-and-forget)
    try
        tp_voxels_for_preload = tp_data_cache[new_tp]
        if !isempty(tp_voxels_for_preload)
            panel1_voxels = tp_voxels_for_preload[1]  # Axial panel data
            for (name, vol) in panel1_voxels
                if name == "CT"
                    InferenceClient.preload_ct_for_nninteractive(Array{Float32,3}(vol))
                    println("[TP Switch] CT preload initiated for $label"); flush(stdout)
                    break
                end
            end
        end
    catch e
        println("[TP Switch] CT preload skipped: $e"); flush(stdout)
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
    println("Refreshing lesion list..."); flush(stdout)
end

function reactToAddAutoPet(data::AddAutoPetEvent, stateObjects::Vector{StateDataFields})
    ai_status_text[] = safe_status_text("[Processing] AI request ($(data.algorithm))...")
    try
        println("Add New Lesion (Auto-PET) triggered with algorithm: $(data.algorithm)"); flush(stdout)
        
        # Log all available volume names for debugging
        tp1_state = stateObjects[1]
        println("[reactToAddAutoPet] Panel 1 volumes: $(join([d.name for d in tp1_state.onScrollData.dataToScroll], ", "))"); flush(stdout)
        
        # Look up volumes by NAME, not index — manualModif auto-inserts at index 2
        ct_vol = nothing
        pet_vol = nothing
        for dat in tp1_state.onScrollData.dataToScroll
            if dat.name == "CT" && ct_vol === nothing
                ct_vol = dat.dat
            elseif dat.name == "PET" && pet_vol === nothing
                pet_vol = dat.dat
            end
        end
        if ct_vol === nothing
            error("CT volume not found by name in panel 1. Available volumes: $(join([d.name for d in tp1_state.onScrollData.dataToScroll], ", ")). No fallbacks allowed.")
        end
        if pet_vol === nothing
            error("PET volume not found by name in panel 1. Available volumes: $(join([d.name for d in tp1_state.onScrollData.dataToScroll], ", ")). No fallbacks allowed.")
        end
        
        seg_vol = nothing
        for dat in tp1_state.onScrollData.dataToScroll
            if dat.name == "Mask" || dat.name == "segmentation"
                seg_vol = dat.dat
                break
            end
        end
        if seg_vol === nothing
            println("[reactToAddAutoPet] WARNING: No 'Mask'/'segmentation' volume found. AI results will not be applied to segmentation volume."); flush(stdout)
        end
        
        # Active lesion ID to assign to predicted voxels: prioritize current_active_lesion_id[]
        active_id = current_active_lesion_id[] > 0 ? current_active_lesion_id[] : tp1_state.valueForMasToSet.value
        if active_id <= 0
            for ts in tp1_state.mainForDisplayObjects.listOfTextSpecifications
                if (ts.isMultiDiscreteMask || ts.name == "Mask" || ts.name == "manualModif") && ts.minAndMaxValue[1] > 0
                    active_id = Int(round(ts.minAndMaxValue[1]))
                    break
                end
            end
        end
        if active_id <= 0
            active_id = 1
        end
        
        # Find user-painted seed points across all panels (Axial, Pure PET, Sagittal, Coronal, Compare)
        painted_pts = CartesianIndex{3}[]
        
        for (p_idx, st) in enumerate(stateObjects)
            # Find what texture this panel is actually configured to paint into
            active_paint_tex = !isempty(st.textureToModifyVec) ? st.textureToModifyVec[1].name : "NONE"
            
            for dat in st.onScrollData.dataToScroll
                # Search for the newly painted scribbles in the texture they actually painted into
                if dat.name == active_paint_tex || dat.name == "manualModif"
                    # For manualModif texture, accept ANY non-zero voxels as scribbles.
                    # For other textures (Mask, segmentation), only match active_id or brush value.
                    is_manual = dat.name == "manualModif"
                    p = if is_manual
                        findall(dat.dat .> 0.0f0)
                    else
                        findall((dat.dat .== Float32(active_id)) .| 
                                (dat.dat .== Float32(st.valueForMasToSet.value)))
                    end
                    if !isempty(p)
                        println("[reactToAddAutoPet] Found $(length(p)) painted voxels for lesion $active_id in panel $p_idx $(dat.name)"); flush(stdout)
                        if p_idx == 3 # Sagittal (Y, Z, X) -> Canonical (X, Y, Z)
                            append!(painted_pts, [CartesianIndex(idx[3], idx[1], idx[2]) for idx in p])
                        elseif p_idx == 4 # Coronal (X, Z, Y) -> Canonical (X, Y, Z)
                            append!(painted_pts, [CartesianIndex(idx[1], idx[3], idx[2]) for idx in p])
                        else # Axial (1, 2, 5) -> Canonical (X, Y, Z)
                            append!(painted_pts, p)
                        end
                    end
                end
            end
        end
        unique!(painted_pts)

        points_vol = zeros(Float32, size(ct_vol))
        if !isempty(painted_pts)
            for idx in painted_pts
                if checkbounds(Bool, points_vol, idx)
                    points_vol[idx] = 1.0f0
                end
            end
            cx = round(Int, mean([p[1] for p in painted_pts]))
            cy = round(Int, mean([p[2] for p in painted_pts]))
            cz = round(Int, mean([p[3] for p in painted_pts]))
        else
            msg = "No painted scribbles found for AI inference. Paint scribbles on the lesion first."
            println("ERROR: $msg"); flush(stdout)
            ai_status_text[] = safe_status_text("[Error] $msg")
            try
                open("/tmp/medeye3d_errors.log", "a") do f
                    println(f, "$(Dates.now()) [reactToAddAutoPet] ERROR: $msg")
                    for (p_idx, st) in enumerate(stateObjects)
                        active_paint_tex = !isempty(st.textureToModifyVec) ? st.textureToModifyVec[1].name : "NONE"
                        println(f, "  Panel $p_idx active painting texture: $active_paint_tex")
                        for dat in st.onScrollData.dataToScroll
                            if dat.name == "manualModif" || dat.name == "Mask" || dat.name == "segmentation"
                                nz_count = count(dat.dat .> 0.0f0)
                                if nz_count > 0
                                    nz_vals = unique(filter(x -> x > 0.0f0, dat.dat))
                                    println(f, "    $(dat.name) contains $nz_count non-zero voxels. Unique values: $nz_vals")
                                else
                                    println(f, "    $(dat.name) is empty (0 non-zero voxels)")
                                end
                            end
                        end
                    end
                    println(f, "---")
                end
            catch; end
            return
        end

        # Capture immutable copies of what the inference worker needs (thread safety)
        algo = data.algorithm
        channel = data.channel
        ct_vol_copy = copy(ct_vol)
        pet_vol_copy = copy(pet_vol)

        # Always use the true CT volume for HELPNet and NNInteractive
        ct_input = ct_vol_copy

        # Pre-extract 0-indexed scribble coordinates for nnInteractive fast path
        # (avoids expensive findall + full-volume allocation in InferenceClient)
        scribble_coords_0idx = [[idx[1]-1, idx[2]-1, idx[3]-1] for idx in painted_pts if checkbounds(Bool, ct_vol, idx)]

        ai_status_text[] = safe_status_text("[Preparing] inference ($(algo))...")
        println("Queuing $(algo) inference job (seed=$cx,$cy,$cz, lesion=$active_id, $(length(painted_pts)) painted points)..."); flush(stdout)

        # Put job on the inference queue — the single persistent worker thread
        # will pick it up and communicate with Docker. No race conditions.
        put!(inference_queue, InferenceJob(
            algo, ct_input, pet_vol_copy, points_vol,
            cx, cy, cz, active_id, seg_vol, channel, scribble_coords_0idx))
    catch e
        err_msg = sprint(showerror, e)
        println("ERROR in reactToAddAutoPet: $err_msg"); flush(stdout)
        println(sprint(showerror, e, catch_backtrace())); flush(stdout)
        ai_status_text[] = safe_status_text("[Error] AI Error: $err_msg")
        try
            open("/tmp/medeye3d_errors.log", "a") do f
                println(f, "$(Dates.now()) reactToAddAutoPet ERROR: $err_msg")
                println(f, sprint(showerror, e, catch_backtrace()))
                println(f, "---")
            end
        catch; end
    end
end

function reactToAIInferenceResult(data::AIInferenceResultEvent, stateObjects::Vector{StateDataFields})
    println("AIInferenceResultEvent received: algorithm=$(data.algorithm), active_id=$(data.active_id), seed=($(data.cx),$(data.cy),$(data.cz))"); flush(stdout)

    if data.mask === nothing
        println("WARNING: AI inference failed or returned nothing."); flush(stdout)
        ai_status_text[] = safe_status_text("[Warning] Inference failed (no mask returned)")
        return
    end

    seg_vol = data.seg_vol
    if seg_vol === nothing
        println("ERROR: No segmentation volume reference available. Cannot apply AI results. No fallbacks allowed."); flush(stdout)
        ai_status_text[] = safe_status_text("[Error] No segmentation volume (Mask) found - cannot apply AI results")
        return
    end

    if size(data.mask) == size(seg_vol)
        seg_vol[data.mask .> 0] .= Float32(data.active_id)
    else
        InferenceClient.insert_patch!(seg_vol, data.mask, data.cx, data.cy, data.cz, label_val=Float32(data.active_id))
    end
    println("$(data.algorithm) segmented $(count(data.mask .> 0)) patch voxels for lesion $(data.active_id) at ($(data.cx), $(data.cy), $(data.cz))."); flush(stdout)
    
    # Compute bone subsegments on the fly for this lesion (only if in bone)
    if global_bone_atlas[] !== nothing
        try
            bone_atlas = global_bone_atlas[]
            lesion_indices = findall(seg_vol .== Float32(data.active_id))
            has_bone_overlap = !isempty(lesion_indices) && any(idx -> checkbounds(Bool, bone_atlas, idx) && bone_atlas[idx] > 0.0f0, lesion_indices)
            if has_bone_overlap
                spacing = (1.5f0, 1.5f0, 2.0f0)
                surf_mask, marr_mask = MedEye3d.BoneSubsegmentation.generate_bone_subsegments(seg_vol, bone_atlas, spacing, data.active_id)
                pts_surf = findall(surf_mask)
                pts_marr = findall(marr_mask)
                bone_subsegments_cache[data.active_id] = (pts_surf, pts_marr)
                println("Generated on-the-fly bone subsegments for lesion $(data.active_id): $(length(pts_surf)) surface, $(length(pts_marr)) marrow"); flush(stdout)
            else
                bone_subsegments_cache[data.active_id] = (CartesianIndex{3}[], CartesianIndex{3}[])
                println("Lesion $(data.active_id) does not overlap bone atlas — skipping bone subsegmentation"); flush(stdout)
            end
        catch e
            println("WARNING: Failed to recalculate bone subsegments for lesion $(data.active_id): $e"); flush(stdout)
        end
    end

    # Update bone surface & marrow textures in all panels
    surf_pts_cached = haskey(bone_subsegments_cache, data.active_id) ? (bone_subsegments_cache[data.active_id][1] isa AbstractArray{<:CartesianIndex} ? bone_subsegments_cache[data.active_id][1] : findall(bone_subsegments_cache[data.active_id][1] .> 0)) : CartesianIndex{3}[]
    marr_pts_cached = haskey(bone_subsegments_cache, data.active_id) ? (bone_subsegments_cache[data.active_id][2] isa AbstractArray{<:CartesianIndex} ? bone_subsegments_cache[data.active_id][2] : findall(bone_subsegments_cache[data.active_id][2] .> 0)) : CartesianIndex{3}[]
    
    for (panel_idx, stateObject) in enumerate(stateObjects)
        surf_indices = if panel_idx == 3
            [CartesianIndex(I[2], I[3], I[1]) for I in surf_pts_cached]
        elseif panel_idx == 4
            [CartesianIndex(I[1], I[3], I[2]) for I in surf_pts_cached]
        else
            surf_pts_cached
        end
        marr_indices = if panel_idx == 3
            [CartesianIndex(I[2], I[3], I[1]) for I in marr_pts_cached]
        elseif panel_idx == 4
            [CartesianIndex(I[1], I[3], I[2]) for I in marr_pts_cached]
        else
            marr_pts_cached
        end
        
        for scrDat in stateObject.onScrollData.dataToScroll
            if scrDat.name == "Bone_Surface"
                fill!(scrDat.dat, 0.0f0)
                if !isempty(surf_indices)
                    scrDat.dat[surf_indices] .= 1.0f0
                end
            elseif scrDat.name == "Bone_Marrow"
                fill!(scrDat.dat, 0.0f0)
                if !isempty(marr_indices)
                    scrDat.dat[marr_indices] .= 1.0f0
                end
            end
        end
    end
    
    # Propagate canonical mask volume to all panels
    for (p_idx, st) in enumerate(stateObjects)
        for scrDat in st.onScrollData.dataToScroll
            if scrDat.name == "manualModif"
                fill!(scrDat.dat, 0.0f0)
            elseif scrDat.name == "segmentation" || scrDat.name == "Mask"
                if p_idx == 3 # Sagittal (Y, Z, X)
                    scrDat.dat .= permutedims(seg_vol, (2, 3, 1))
                elseif p_idx == 4 # Coronal (X, Z, Y)
                    scrDat.dat .= permutedims(seg_vol, (1, 3, 2))
                else # Axial (1, 2, 5)
                    scrDat.dat .= seg_vol
                end
            end
        end
    end

    # Synchronize tp_data_cache
    tp_idx = current_tp_index[]
    if haskey(tp_data_cache, tp_idx)
        tp_voxels = tp_data_cache[tp_idx]
        for (p_idx, panel_data) in enumerate(tp_voxels)
            for entry in panel_data
                if entry[1] == "Mask" || entry[1] == "manualModif" || entry[1] == "segmentation"
                    for st_dat in stateObjects[p_idx].onScrollData.dataToScroll
                        if st_dat.name == entry[1]
                            entry[2] .= st_dat.dat
                            break
                        end
                    end
                end
            end
        end
    end

    # Ensure mask uniform displays the active lesion
    for stateObject in stateObjects
        for textSpec in stateObject.mainForDisplayObjects.listOfTextSpecifications
            if textSpec.name == "Mask" || textSpec.name == "segmentation"
                textSpec.minAndMaxValue = Float32.([data.active_id, data.active_id])
                ModernGL.glUseProgram(stateObject.mainForDisplayObjects.shader_program)
                Uniforms.coontrolMinMaxUniformVals(textSpec)
            elseif textSpec.name == "manualModif"
                textSpec.minAndMaxValue = Float32.([0.0, 1000.0])
                ModernGL.glUseProgram(stateObject.mainForDisplayObjects.shader_program)
                Uniforms.coontrolMinMaxUniformVals(textSpec)
            end
        end
    end

    # Compute center of the actual segmentation result (not the seed point)
    seg_indices = findall(seg_vol .== Float32(data.active_id))
    if !isempty(seg_indices)
        center_x = round(Int, mean(i[1] for i in seg_indices))
        center_y = round(Int, mean(i[2] for i in seg_indices))
        center_z = round(Int, mean(i[3] for i in seg_indices))
        println("[AI Result] Centering on segmentation centroid: ($center_x, $center_y, $center_z) from $(length(seg_indices)) voxels"); flush(stdout)
    else
        center_x, center_y, center_z = data.cx, data.cy, data.cz
        println("[AI Result] No segmented voxels found, using seed: ($center_x, $center_y, $center_z)"); flush(stdout)
    end
    # Jump panels to segmentation center
    targets = [(1, center_z), (2, center_z), (3, center_x), (4, center_y)]
    if length(stateObjects) >= 5
        push!(targets, (5, center_z))
    end
    for (p_idx, target_sl) in targets
        if p_idx <= length(stateObjects)
            st = stateObjects[p_idx]
            max_sl = st.onScrollData.slicesNumber
            st.currentDisplayedSlice = clamp(target_sl, 1, max_sl)
        end
    end

    # Update all panel textures via on_next! multiple dispatch (scroll with 0 = re-render current slice)
    old_sw = stateObjects[1].switchIndex
    for p in 1:length(stateObjects)
        if sum(abs.(stateObjects[p].calcDimsStruct.mainImageQuadVert)) > 0.01f0
            stateObjects[1].switchIndex = p
            reactToScroll(0, stateObjects)
        end
    end
    stateObjects[1].switchIndex = old_sw

    # Update status label
    voxel_count = count(data.mask .> 0)
    ai_status_text[] = safe_status_text("[Success] Done ($(voxel_count) voxels, lesion $(data.active_id))")
end
function reactToSyncMissing(data::SyncMissingEvent, stateObjects::Vector{StateDataFields})
    println("Sync Missing Lesions across TPs triggered."); flush(stdout)
    if length(stateObjects) < 2
        println("WARNING: Need at least 2 time points to sync missing lesions."); flush(stdout)
        return
    end
    
    tp1_state = stateObjects[1]
    tp2_state = stateObjects[2]
    
    # For now, just sync the current position. A full sync would iterate over all unique values in tp1_seg.
    pos = tp1_state.lastRecordedMousePosition
    
    tp2_ct = tp2_state.mainForDisplayObjects.listOfTextSpecifications[1].imageTexture
    tp2_pet = tp2_state.mainForDisplayObjects.listOfTextSpecifications[2].imageTexture
    tp2_seg = tp2_state.mainForDisplayObjects.listOfTextSpecifications[3].imageTexture
    
    println("Running HelpNet on TP2 for missing lesion at $pos..."); flush(stdout)
    mask = InferenceClient.run_helpnet_inference(tp2_ct, tp2_pet, pos[1], pos[2], pos[3])
    if mask !== nothing
        InferenceClient.insert_patch!(tp2_seg, mask, pos[1], pos[2], pos[3])
        LesionAssociation.map_link("TP1", "TP2", "SyncedLesion")
        println("Successfully synced and mapped lesion to TP2."); flush(stdout)
    end
end

function reactToGenManual(data::GenManualEvent, stateObjects::Vector{StateDataFields})
    println("Bone subsegmentation triggered for lesion $(data.lesion_id)"); flush(stdout)
    tp1_state = stateObjects[1]
    
    seg_vol = tp1_state.mainForDisplayObjects.listOfTextSpecifications[3].imageTexture
    ct_vol = tp1_state.mainForDisplayObjects.listOfTextSpecifications[2].imageTexture
    
    bone_atlas = Float32.(ct_vol .> 150.0f0)
    spacing = (1.5f0, 1.5f0, 2.0f0)
    
    try
        surf, marr = MedEye3d.BoneSubsegmentation.generate_bone_subsegments(seg_vol, bone_atlas, spacing, data.lesion_id)
        
        tp_idx = current_tp_index[]
        if haskey(tp_data_cache, tp_idx)
            tp_voxels = tp_data_cache[tp_idx]
            
            # Since tp_voxels is a Vector{Vector{Any}} (one for each panel)
            for (panel_idx, panel_data) in enumerate(tp_voxels)
                # Create masks based on the panel's data dimension
                # Panel 1 and 5 (axial), Panel 3 (sagittal), Panel 4 (coronal)
                # For simplicity, we create the full 3D volumes in axial layout first
                surf_vol = zeros(Float32, size(seg_vol))
                marr_vol = zeros(Float32, size(seg_vol))
                surf_vol[surf] .= 1.0f0
                marr_vol[marr] .= 1.0f0
                
                # Transform to panel's orientation if needed
                if panel_idx == 3 # Sagittal (Y, Z, X)
                    surf_vol = permutedims(surf_vol, (2, 3, 1))
                    marr_vol = permutedims(marr_vol, (2, 3, 1))
                elseif panel_idx == 4 # Coronal (X, Z, Y)
                    surf_vol = permutedims(surf_vol, (1, 3, 2))
                    marr_vol = permutedims(marr_vol, (1, 3, 2))
                end
                
                # Remove existing if any
                filter!(v -> v[1] != "Bone_Surface" && v[1] != "Bone_Marrow", panel_data)
                
                # Append
                push!(panel_data, ("Bone_Surface", surf_vol))
                push!(panel_data, ("Bone_Marrow", marr_vol))
            end
            
            println("Bone subsegments generated and saved as separate label maps."); flush(stdout)
            
            # Re-initialize the panels to register the new textures
            num_panels = min(4, length(stateObjects))
            for i in 1:num_panels
                _load_tp_into_panel!(stateObjects, tp_voxels, i)
            end
            
            old_idx = stateObjects[1].switchIndex
            for i in 1:num_panels
                stateObjects[1].switchIndex = i
                reactToScroll(0, stateObjects, false)
            end
            stateObjects[1].switchIndex = old_idx
        else
            error("tp_data_cache not available. Cannot generate bone subsegments outside interactive mode. No fallbacks allowed.")
        end
        
    catch e
        println("ERROR: Failed to generate bone subsegments: $e"); flush(stdout)
    end
end

function reactToMapLink(data::MapLinkEvent, stateObjects::Vector{StateDataFields})
    println("Map Link triggered. Linking lesions..."); flush(stdout)
    # Normally we would link the currently active lesion from TP1 to TP2
    # This acts as a manual override matching.
    if length(stateObjects) > 1
        LesionAssociation.map_link("TP1", "TP2", data.lesion_id)
        println("Lesion $(data.lesion_id) successfully mapped between TP1 and TP2"); flush(stdout)
    end
end

function reactToAutoRunPreprocess(data::AutoRunPreprocessEvent, stateObjects::Vector{StateDataFields})
    println("Auto-run preprocessing toggled to $(data.active)"); flush(stdout)
end

function reactToRunPreprocess(data::RunPreprocessEvent, stateObjects::Vector{StateDataFields})
    println("Full Preprocessing triggered."); flush(stdout)
end

function reactToShowBoneMask(data::ShowBoneMaskEvent, stateObjects::Vector{StateDataFields})
    println("Show Bone Mask toggled to $(data.active)"); flush(stdout)
    for stateObject in stateObjects
        ModernGL.glUseProgram(stateObject.mainForDisplayObjects.shader_program)
        for textSpec in stateObject.mainForDisplayObjects.listOfTextSpecifications
            if textSpec.name == "Bone_Mask" || textSpec.name == "bone_mask" || textSpec.name == "bone" || textSpec.name == "Organ_Mask" || textSpec.name == "organ_mask" || textSpec.name == "Bone_Surface" || textSpec.name == "Bone_Marrow"
                textSpec.isVisible = data.active
                Uniforms.setTextureVisibility(textSpec.isVisible, textSpec.uniforms)
            end
        end
    end
    old_sw = stateObjects[1].switchIndex
    for p in 1:length(stateObjects)
        if sum(abs.(stateObjects[p].calcDimsStruct.mainImageQuadVert)) > 0.01f0
            stateObjects[1].switchIndex = p
            ReactToScroll.reactToScroll(0, stateObjects, false)
        end
    end
    stateObjects[1].switchIndex = old_sw
end

const MASK_BACKUP = Dict{UInt64, Array{Float32, 3}}()

function reactToShowMaskLayer(data::ShowMaskLayerEvent, stateObjects::Vector{StateDataFields})
    println("reactToShowMaskLayer: layer=$(data.layer) active=$(data.active) num_panels=$(length(stateObjects))")
    flush(stdout)
    
    tex_target = if data.layer == 1
        "Mask"
    elseif data.layer == 2
        "Bone_Surface"
    elseif data.layer == 3
        "Bone_Marrow"
    else
        ""
    end
    
    toggled_count = 0
    for (si, state) in enumerate(stateObjects)
        ModernGL.glUseProgram(state.mainForDisplayObjects.shader_program)
        for textSpec in state.mainForDisplayObjects.listOfTextSpecifications
            if (tex_target == "Mask" && (textSpec.name == "Mask" || textSpec.name == "manualModif" || textSpec.name == "segmentation")) ||
               (textSpec.name == tex_target)
                textSpec.isVisible = data.active
                Uniforms.setTextureVisibility(textSpec.isVisible, textSpec.uniforms)
                toggled_count += 1
                println("  Panel $si: set isVisible=$(data.active) for texture '$(textSpec.name)'")
            end
        end
        TextureManag.activateTextures(state.mainForDisplayObjects.listOfTextSpecifications)
    end
    println("  Toggled $toggled_count textures total for layer=$(data.layer)")
    flush(stdout)
    
    # If bone surface or marrow is toggled, also sync dataToScroll buffer directly
    if data.layer == 2 || data.layer == 3
        cur_lid = (current_active_lesion_id[] > 0) ? current_active_lesion_id[] : stateObjects[1].valueForMasToSet.value
        println("  Bone data sync: cur_lid=$cur_lid, in_cache=$(haskey(bone_subsegments_cache, cur_lid))")
        flush(stdout)
        for (panel_idx, stateObject) in enumerate(stateObjects)
            for scrDat in stateObject.onScrollData.dataToScroll
                if (data.layer == 2 && scrDat.name == "Bone_Surface") || (data.layer == 3 && scrDat.name == "Bone_Marrow")
                    if !data.active
                        fill!(scrDat.dat, 0.0f0)
                    elseif cur_lid > 0 && haskey(bone_subsegments_cache, cur_lid)
                        raw_surf, raw_marr = bone_subsegments_cache[cur_lid]
                        pts = (data.layer == 2) ? (raw_surf isa AbstractArray{<:CartesianIndex} ? raw_surf : findall(raw_surf .> 0)) :
                                                  (raw_marr isa AbstractArray{<:CartesianIndex} ? raw_marr : findall(raw_marr .> 0))
                        
                        # Use canonical indices matching reactToSyncLesion and reactToActiveLesionChanged
                        indices = if panel_idx == 3 # Sagittal (Y, Z, X)
                            [CartesianIndex(I[2], I[3], I[1]) for I in pts]
                        elseif panel_idx == 4 # Coronal (X, Z, Y)
                            [CartesianIndex(I[1], I[3], I[2]) for I in pts]
                        else # Axial (X, Y, Z)
                            pts
                        end
                        fill!(scrDat.dat, 0.0f0)
                        if !isempty(indices)
                            scrDat.dat[indices] .= 1.0f0
                        end
                    end
                end
            end
        end
    end
    
    # Re-render all panels
    old_idx = stateObjects[1].switchIndex
    for idx in 1:length(stateObjects)
        if sum(abs.(stateObjects[idx].calcDimsStruct.mainImageQuadVert)) > 0.01f0
            stateObjects[1].switchIndex = idx
            try
                ReactToScroll.reactToScroll(0, stateObjects, false)
            catch e
                println("reactToScroll ERROR for panel $idx during visibility toggle: $e")
                println(sprint(showerror, e, catch_backtrace()))
                flush(stdout)
            end
        end
    end
    stateObjects[1].switchIndex = old_idx
end

function reactToSaveMRB(data::SaveMRBEvent, stateObjects::Vector{StateDataFields})
    println("Save MRB triggered."); flush(stdout)
end

export reactToToggleMoveLesionMode
function reactToToggleMoveLesionMode(data::ToggleMoveLesionModeEvent, stateObjects::Vector{StateDataFields})
    println("Move Lesion Mode toggled to $(data.active)"); flush(stdout)
    for state in stateObjects
        state.moveLesionMode = data.active
    end
end

end
