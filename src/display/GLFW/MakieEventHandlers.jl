module MakieEventHandlers
using ...MakieEvents
using ...StructsManag
using ...ForDisplayStructs
using ...DataStructs
using ...ChangePlane
using ...ReactToScroll
using ...DisplayWords
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
# ModernGL removed — Vulkan UBO updates happen in consumer loop via update_ubo!
# Uniforms module no longer needed — TextureSpec fields are read directly by UBO packer
using Observables

# Debug flag: set to true to enable verbose bench/bone logging in hot paths
const DEBUG_VERBOSE = Ref(false)

# AI status Observable — LesionMetadataWindow reads this for the GUI label
const ai_status_text = Observable{String}("Ready")

# Cursor info Observables — updated from on_next!(MouseStruct) via reactToMouseDrag
const cursor_info_text = Observable{String}("")      # "HU: 45 | SUV: 3.2 | femur (L5) | [Ax] Sl:163"
const cursor_study_text = Observable{String}("")     # "PET TP0" or "L: PET TP0 | R: PET TP3"
export cursor_info_text, cursor_study_text

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
    target = round(T, lesion_id)
    sx, sy, sz = size(dat)
    sum_x = 0; sum_y = 0; sum_z = 0; n = 0
    @inbounds for z in 1:sz, y in 1:sy, x in 1:sx
        v = dat[x, y, z]
        if v == target || abs(Float32(v) - lesion_id) < 0.1f0
            sum_x += x
            sum_y += y
            sum_z += z
            n += 1
        end
    end
    if n == 0
        return nothing
    end
    return [round(Int, sum_x / n), round(Int, sum_y / n), round(Int, sum_z / n)]
end

"""
Fast single-pass accumulator to precompute centroids for ALL unique lesion IDs in a mask volume.
Populates lesion_centroids_cache with (tp_idx, lid), (node_name, lid), and lid keys.
"""
function precompute_mask_centroids!(mask_vol::AbstractArray{T, 3}, tp_idx::Int, node_name::String="") where T
    sx, sy, sz = size(mask_vol)
    sums_x = Dict{Int, Int}()
    sums_y = Dict{Int, Int}()
    sums_z = Dict{Int, Int}()
    counts = Dict{Int, Int}()
    
    @inbounds for z in 1:sz, y in 1:sy, x in 1:sx
        v = Int(round(mask_vol[x, y, z]))
        if v > 0
            sums_x[v] = get(sums_x, v, 0) + x
            sums_y[v] = get(sums_y, v, 0) + y
            sums_z[v] = get(sums_z, v, 0) + z
            counts[v] = get(counts, v, 0) + 1
        end
    end
    
    for (lid, n) in counts
        c = [round(Int, sums_x[lid] / n), round(Int, sums_y[lid] / n), round(Int, sums_z[lid] / n)]
        lesion_centroids_cache[(tp_idx, lid)] = c
        if !isempty(node_name)
            lesion_centroids_cache[(node_name, lid)] = c
        end
        if tp_idx == current_tp_index[]
            lesion_centroids_cache[lid] = c
        end
    end
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
    panel_indices = Int[]
    for (idx, stateObject) in enumerate(stateObjects)
        old_scroll = stateObject.onScrollData.dataToScrollDims
        new_scroll = DataToScrollDims(imageSize=old_scroll.imageSize, voxelSize=old_scroll.voxelSize, dimensionToScroll=dim)
        
        stateObject.lastRecordedMousePosition = CartesianIndex(
            max(1, round(Int, old_scroll.imageSize[1] / 2)),
            max(1, round(Int, old_scroll.imageSize[2] / 2)),
            max(1, round(Int, old_scroll.imageSize[3] / 2))
        )
        
        ChangePlane.processKeysInfo(Identity(new_scroll), stateObject, dummy_kb, false)
        
        if compare_mode[]
            if idx == 1
                updateQuadVertices!(stateObject, :LeftHalf)
            elseif idx == 5
                updateQuadVertices!(stateObject, :RightHalf)
            elseif idx in (2, 3, 4)
                updateQuadVertices!(stateObject, :Hidden)
            end
        end
        push!(panel_indices, idx)
    end
    # Batch texture upload for all panels at once
    ReactToScroll.reactToScrollMultiPanel!(panel_indices, stateObjects)
end

function updateQuadVertices!(stateObject::StateDataFields, layout::Symbol)
    calcDimStruct = stateObject.calcDimsStruct
    
    if layout == :Hidden
        res = zeros(Float32, 32)
        w_res = zeros(Float32, 32)
        stateObject.calcDimsStruct = Setfield.setproperties(calcDimStruct, (
            mainImageQuadVert = res, 
            mainQuadVertSize = sizeof(res),
            wordsImageQuadVert = w_res,
            wordsQuadVertSize = sizeof(w_res)
        ))
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
end

const compare_mode = Ref(false)
const compare_right_tp = Ref(-1)  # TP index shown in right panel (panel 5)
const tp_switched = Observable{Int}(0)

"""Force direct texture upload for a panel — bypasses scroll pipeline entirely."""
function _force_texture_upload!(stateObjects::Vector{StateDataFields}, panel_idx::Int)
    panelState = stateObjects[panel_idx]
    dimToScroll = panelState.onScrollData.dimensionToScroll
    lastSlice = panelState.onScrollData.slicesNumber
    if lastSlice < 1
        println("  [COMPARE-DBG] panel $panel_idx: slicesNumber=$lastSlice, SKIPPING upload"); flush(stdout)
        return
    end
    current = clamp(panelState.currentDisplayedSlice, 1, lastSlice)
    
    singleSlDat = panelState.onScrollData.dataToScroll |>
        (scrDat) -> map(threeDimDat -> threeToTwoDimm(threeDimDat.type, Int64(current), dimToScroll, threeDimDat), scrDat) |>
        (twoDimList) -> SingleSliceDat(listOfDataAndImageNames=twoDimList, sliceNumber=current, textToDisp=getTextForCurrentSlice(panelState.onScrollData, Int32(current)))
    
    n_uploaded = 0
    for updateDat in singleSlDat.listOfDataAndImageNames
        findList = findall((texSpec) -> texSpec.name == updateDat.name, panelState.mainForDisplayObjects.listOfTextSpecifications)
        if !isempty(findList)
            texSpec = panelState.mainForDisplayObjects.listOfTextSpecifications[findList[1]]
            # GPU zoom/pan: upload raw unzoomed data — zoom/pan applied by vertex shader
            # Safety: verify data dimensions fit within the allocated texture
            actual_w = size(updateDat.dat, 1)
            actual_h = size(updateDat.dat, 2)
            tex_w = Int(panelState.calcDimsStruct.imageTextureWidth)
            tex_h = Int(panelState.calcDimsStruct.imageTextureHeight)
            if actual_w <= tex_w && actual_h <= tex_h
                TextureManag.updateTexture(updateDat.type, updateDat.dat, texSpec, 0, 0, panelState.calcDimsStruct.imageTextureWidth, panelState.calcDimsStruct.imageTextureHeight)
                n_uploaded += 1
            else
                println("  [COMPARE-DBG] SKIPPING texture upload for '$(updateDat.name)' on panel $panel_idx: data=$(actual_w)x$(actual_h) > texture=$(tex_w)x$(tex_h)"); flush(stdout)
            end
        end
    end
    
    panelState.currentlyDispDat = singleSlDat
    panelState.currentDisplayedSlice = current
    panelState.isSliceChanged = true
    println("  [COMPARE-DBG] panel $panel_idx: uploaded $n_uploaded textures at slice $current (dimToScroll=$dimToScroll, slicesNumber=$lastSlice)"); flush(stdout)
end

function reactToCompareTimePoints(data::CompareTimePointsEvent, stateObjects::Vector{StateDataFields})
    if length(stateObjects) >= 5
        compare_mode[] = data.compare
        if data.compare
            # Load the NEXT TP into panel 5
            tp_indices = sort(collect(keys(tp_labels)))
            if !isempty(tp_indices)
                cur_pos = findfirst(==(current_tp_index[]), tp_indices)
                cur_pos = cur_pos === nothing ? 1 : cur_pos
                next_pos = mod1(cur_pos + 1, length(tp_indices))
                right_tp = tp_indices[next_pos]
                compare_right_tp[] = right_tp
                
                # Load right TP data into panel 5 using _load_tp_from_entry!
                entry = get_or_load_tp_data(right_tp)
                if entry !== nothing
                    rm = entry.mask_f32
                    rs = get_existing_bone_array(stateObjects[5], "Bone_Surface")
                    rmr = get_existing_bone_array(stateObjects[5], "Bone_Marrow")
                    _load_tp_from_entry!(stateObjects, entry, 5; mask_f32=rm, bone_s_f32=rs, bone_m_f32=rmr)
                end
            end

            # 2-pane view: panel 1 on left, panel 5 on right
            updateQuadVertices!(stateObjects[1], :LeftHalf)
            updateQuadVertices!(stateObjects[5], :RightHalf)
            updateQuadVertices!(stateObjects[2], :Hidden)
            updateQuadVertices!(stateObjects[3], :Hidden)
            updateQuadVertices!(stateObjects[4], :Hidden)

            left_label = get(tp_labels, current_tp_index[], "TP $(current_tp_index[])")
            right_label = get(tp_labels, compare_right_tp[], "TP $(compare_right_tp[])")
            println("Compare mode ON: Left=$left_label, Right=$right_label"); flush(stdout)
            
            # Force direct texture upload for both visible panels
            _force_texture_upload!(stateObjects, 1)
            _force_texture_upload!(stateObjects, 5)
            
            # If there's an active lesion, set mask filter uniforms
            if current_active_lesion_id[] > 0
                try
                    reactToSyncLesion(SyncLesionEvent(current_active_lesion_id[]), stateObjects)
                catch e
                    println("WARNING: reactToSyncLesion failed during compare-ON: $e"); flush(stdout)
                end
            end
        else
            compare_right_tp[] = -1
            # Reload current active TP into all 4 panels using _load_tp_from_entry!
            entry = get_or_load_tp_data(current_tp_index[])
            if entry !== nothing
                m = entry.mask_f32
                s = get_existing_bone_array(stateObjects[1], "Bone_Surface")
                mr = get_existing_bone_array(stateObjects[1], "Bone_Marrow")
                num_panels = min(4, length(stateObjects))
                for i in 1:num_panels
                    _load_tp_from_entry!(stateObjects, entry, i; mask_f32=m, bone_s_f32=s, bone_m_f32=mr)
                end
            end
            # All TPs pre-loaded before GUI launch — no eviction needed

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

            # Clear display data to force full texture re-upload
            for i in 1:4
                stateObjects[i].currentlyDispDat = SingleSliceDat(sliceNumber=0)
            end
            println("Compare mode OFF: restored 4-pane view for TP $(current_tp_index[])"); flush(stdout)
            
            # Force direct texture upload for all 4 visible panels
            for i in 1:4
                _force_texture_upload!(stateObjects, i)
            end
            
            # If there's an active lesion, set mask filter uniforms
            if current_active_lesion_id[] > 0
                try
                    reactToSyncLesion(SyncLesionEvent(current_active_lesion_id[]), stateObjects)
                catch e
                    println("WARNING: reactToSyncLesion failed during compare-OFF: $e"); flush(stdout)
                end
            end
        end
        tp_switched[] = tp_switched[] + 1
    end
end

function reactToShowSingleLesion(data::ShowSingleLesionEvent, stateObjects::Vector{StateDataFields})
    changed = false
    for stateObject in stateObjects
        for textSpec in stateObject.mainForDisplayObjects.listOfTextSpecifications
            if textSpec.isMultiDiscreteMask || textSpec.name == "Mask"
                # Clear allowed IDs filter
                textSpec.allowedIDs = Float32[]
                if data.lesion_id == 0
                    textSpec.minAndMaxValue = Float32.([1.0, 1000.0])
                else
                    textSpec.minAndMaxValue = Float32.([data.lesion_id, data.lesion_id])
                end
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
                # UBO update happens in consumer loop
            end
        end
    end
    println("Updated windowing for $(data.modality): [$(data.min_val), $(data.max_val)]"); flush(stdout)
end

function reactToPetBlend(data::PetBlendEvent, stateObjects::Vector{StateDataFields})
    for state in stateObjects
        for tex in state.mainForDisplayObjects.listOfTextSpecifications
            # Update nuclear overlay contribution (PET/SPECT, not the pure PET main image panel)
            if tex.isNuclearMask && !tex.isMainImage
                tex.maskContribution = clamp(data.weight, 0.0f0, 1.0f0)
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
                    # UBO update happens in consumer loop
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
    if DEBUG_VERBOSE[]; println("Brush size updated to $(data.size)"); flush(stdout); end
end

const tp_node_names = Dict{Int, String}()

function get_node_name_for_tp(tp_idx::Int)::String
    if haskey(tp_node_names, tp_idx)
        return tp_node_names[tp_idx]
    end
    @warn "No node name for TP $tp_idx — tp_node_names not populated from HDF5"
    return "Unknown_TP_$tp_idx"
end

const current_active_lesion_id = Ref(0)

"""
Retrieve precomputed bone subsegments from cache (loaded from Bone_Subsegments_0.h5 at startup).
NO on-the-fly computation — all bone data must be precomputed via preprocessing.
If a lesion has no cache entry, it is not a bone lesion (filtered by 5% Skellytour overlap threshold).
Returns (surf_pts::Vector{CartesianIndex{3}}, marr_pts::Vector{CartesianIndex{3}})
"""
function _get_or_compute_bone_subseg(stateObject, target_id::Int, panel_tp::Int)
    if target_id <= 0
        return (CartesianIndex{3}[], CartesianIndex{3}[])
    end
    
    node_name = get_node_name_for_tp(panel_tp)
    
    # Check precomputed cache (loaded from Bone_Subsegments_0.h5 at startup)
    cached = if haskey(bone_subsegments_cache, (panel_tp, target_id))
        if DEBUG_VERBOSE[]; println("  [BONE] Cache HIT by (tp=$panel_tp, lid=$target_id)"); flush(stdout); end
        bone_subsegments_cache[(panel_tp, target_id)]
    elseif haskey(bone_subsegments_cache, (node_name, target_id))
        if DEBUG_VERBOSE[]; println("  [BONE] Cache HIT by (node=$node_name, lid=$target_id)"); flush(stdout); end
        bone_subsegments_cache[(node_name, target_id)]
    else
        if DEBUG_VERBOSE[]; println("  [BONE] Cache MISS for lid=$target_id tp=$panel_tp node=$node_name"); flush(stdout); end
        nothing
    end
    
    if cached !== nothing
        if cached === :computing
            return (CartesianIndex{3}[], CartesianIndex{3}[])
        end
        raw_surf, raw_marr = cached
        surf_res = raw_surf isa AbstractArray{<:CartesianIndex} ? raw_surf : findall(raw_surf .> 0)
        marr_res = raw_marr isa AbstractArray{<:CartesianIndex} ? raw_marr : findall(raw_marr .> 0)
        if DEBUG_VERBOSE[]; println("  [BONE] Result: $(length(surf_res)) surf, $(length(marr_res)) marr"); flush(stdout); end
        return (surf_res, marr_res)
    end
    
    # No precomputed data → not a bone lesion (filtered by 5% Skellytour overlap in preprocessing)
    # Cache the empty result to avoid repeated lookups
    bone_subsegments_cache[(panel_tp, target_id)] = (CartesianIndex{3}[], CartesianIndex{3}[])
    return (CartesianIndex{3}[], CartesianIndex{3}[])
end

function reactToSyncLesion(data::SyncLesionEvent, stateObjects::Vector{StateDataFields})
    t_total = time_ns()
    changed = false

    # 1. Update OpenGL visibility constants for masks
    panel5_lesion_id = data.lesion_id
    panel5_all_ids = Int[]
    if compare_mode[] && length(stateObjects) >= 5 && data.lesion_id > 0
        try
            left_node = get_node_name_for_tp(current_tp_index[])
            right_node = get_node_name_for_tp(compare_right_tp[])
            match_mod = isdefined(Main, :MedEye3d) && isdefined(Main.MedEye3d, :LesionAssociation) ? 
                        Main.MedEye3d.LesionAssociation : nothing
            if match_mod !== nothing
                matched_ids = match_mod.find_cross_tp_lesion(left_node, data.lesion_id, right_node)
                if !isempty(matched_ids)
                    panel5_all_ids = matched_ids
                    panel5_lesion_id = matched_ids[1]
                end
            end
        catch e
            println("WARNING: Error finding cross-TP lesion: $e"); flush(stdout)
        end
    end

    for (idx, stateObject) in enumerate(stateObjects)
        target_id = (idx == 5 && compare_mode[]) ? panel5_lesion_id : data.lesion_id
        for textSpec in stateObject.mainForDisplayObjects.listOfTextSpecifications
            if textSpec.name == "Mask" || textSpec.name == "segmentation"
                if idx == 5 && compare_mode[] && !isempty(panel5_all_ids)
                    textSpec.allowedIDs = Float32.(panel5_all_ids)
                else
                    textSpec.allowedIDs = Float32[]
                    if target_id > 0
                        textSpec.minAndMaxValue = Float32.([target_id, target_id])
                    else
                        textSpec.minAndMaxValue = Float32.([1.0, 1000.0])
                    end
                end
            elseif textSpec.name == "manualModif"
                textSpec.minAndMaxValue = Float32.([0.0, 1000.0])
                textSpec.allowedIDs = Float32[]
            end
        end
    end

    # 1b. Update bone subseg 3D arrays for all panels
    has_any_bone_data = false
    if data.lesion_id > 0
        for (panel_idx, stateObject) in enumerate(stateObjects)
            panel_tp = (panel_idx == 5 && compare_mode[]) ? compare_right_tp[] : current_tp_index[]
            panel_lid = (panel_idx == 5 && compare_mode[]) ? panel5_lesion_id : data.lesion_id

            panel_surf_pts, panel_marr_pts = try
                _get_or_compute_bone_subseg(stateObject, panel_lid, panel_tp)
            catch e
                (CartesianIndex{3}[], CartesianIndex{3}[])
            end
            surf_indices = if panel_idx == 3  # Sagittal (Y, Z, X)
                [CartesianIndex(I[2], I[3], I[1]) for I in panel_surf_pts]
            elseif panel_idx == 4  # Coronal (X, Z, Y)
                [CartesianIndex(I[1], I[3], I[2]) for I in panel_surf_pts]
            else  # Axial (panels 1, 2, 5)
                panel_surf_pts
            end
            marr_indices = if panel_idx == 3
                [CartesianIndex(I[2], I[3], I[1]) for I in panel_marr_pts]
            elseif panel_idx == 4
                [CartesianIndex(I[1], I[3], I[2]) for I in panel_marr_pts]
            else
                panel_marr_pts
            end

            if !isempty(surf_indices) || !isempty(marr_indices)
                has_any_bone_data = true
            end

            for scrDat in stateObject.onScrollData.dataToScroll
                if scrDat.name == "Bone_Surface"
                    if haskey(last_bone_surf_indices, panel_idx) && !isempty(last_bone_surf_indices[panel_idx])
                        scrDat.dat[last_bone_surf_indices[panel_idx]] .= 0.0f0
                    end
                    if !isempty(surf_indices)
                        scrDat.dat[surf_indices] .= 1.0f0
                    end
                    last_bone_surf_indices[panel_idx] = surf_indices
                elseif scrDat.name == "Bone_Marrow"
                    if haskey(last_bone_marr_indices, panel_idx) && !isempty(last_bone_marr_indices[panel_idx])
                        scrDat.dat[last_bone_marr_indices[panel_idx]] .= 0.0f0
                    end
                    if !isempty(marr_indices)
                        scrDat.dat[marr_indices] .= 1.0f0
                    end
                    last_bone_marr_indices[panel_idx] = marr_indices
                end
            end
        end

        # Ensure bone textures are visible in the shader when bone data exists
        if has_any_bone_data
            for stateObject in stateObjects
                for textSpec in stateObject.mainForDisplayObjects.listOfTextSpecifications
                    if textSpec.name == "Bone_Surface" || textSpec.name == "Bone_Marrow"
                        textSpec.isVisible = true
                    end
                end
            end
        end
    end

    # 2. Get canonical center
    panel_tp_cur = current_tp_index[]
    canonical_center = if data.lesion_id > 0
        if haskey(lesion_centroids_cache, (panel_tp_cur, data.lesion_id))
            lesion_centroids_cache[(panel_tp_cur, data.lesion_id)]
        elseif haskey(lesion_centroids_cache, (get_node_name_for_tp(panel_tp_cur), data.lesion_id))
            lesion_centroids_cache[(get_node_name_for_tp(panel_tp_cur), data.lesion_id)]
        elseif haskey(lesion_centroids_cache, data.lesion_id)
            lesion_centroids_cache[data.lesion_id]
        else
            # on-the-fly computation if cache miss
            cc = nothing
            for (si, stateObject) in enumerate(stateObjects)
                for (scrIdx, scrDat) in enumerate(stateObject.onScrollData.dataToScroll)
                    texSpec = stateObject.mainForDisplayObjects.listOfTextSpecifications[scrIdx]
                    if (texSpec.name == "Mask" || texSpec.name == "manualModif") && stateObject.onScrollData.dimensionToScroll == 3
                        cc = find_lesion_center(scrDat.dat, Float32(data.lesion_id))
                        if cc !== nothing
                            lesion_centroids_cache[(panel_tp_cur, data.lesion_id)] = cc
                            break
                        end
                    end
                end
                cc !== nothing && break
            end
            cc
        end
    else
        nothing
    end

    # 3. Emulate Right Click behavior to jump panels exactly as right click does
    if canonical_center !== nothing
        origX, origY, origZ = canonical_center[1], canonical_center[2], canonical_center[3]

        for i in 1:length(stateObjects)
            if i == 1 || i == 2 || i == 5
                stateObjects[i].lastRecordedMousePosition = CartesianIndex(origX, origY, origZ)
            elseif i == 3
                stateObjects[i].lastRecordedMousePosition = CartesianIndex(origY, origZ, origX)
            else
                stateObjects[i].lastRecordedMousePosition = CartesianIndex(origX, origZ, origY)
            end
        end

        targets = [(1, origZ), (2, origZ), (3, origX), (4, origY)]
        if length(stateObjects) >= 5
            push!(targets, (5, origZ))
        end

        for (p_idx, targetSlice) in targets
            if p_idx <= length(stateObjects)
                otherState = stateObjects[p_idx]
                lastSlice = max(1, otherState.onScrollData.slicesNumber)
                newSlice = clamp(targetSlice, 1, lastSlice)
                
                # Create slice dat (forces evaluation)
                singleSlDat = otherState.onScrollData.dataToScroll |>
                    (scrDat) -> map(threeDimDat -> threeToTwoDimm(threeDimDat.type, Int64(newSlice), otherState.onScrollData.dimensionToScroll, threeDimDat), scrDat) |>
                    (twoDimList) -> SingleSliceDat(listOfDataAndImageNames=twoDimList, sliceNumber=newSlice, textToDisp=getTextForCurrentSlice(otherState.onScrollData, Int32(newSlice)))
                
                # Fast upload directly to OpenGL (skipping CPU sync scroll logic)
                for updateDat in singleSlDat.listOfDataAndImageNames
                    findList = findall((texSpec) -> texSpec.name == updateDat.name, otherState.mainForDisplayObjects.listOfTextSpecifications)
                    if !isempty(findList)
                        texSpec = otherState.mainForDisplayObjects.listOfTextSpecifications[findList[1]]
                        # GPU zoom/pan: upload raw unzoomed data — zoom/pan applied by vertex shader
                        TextureManag.updateTexture(updateDat.type, updateDat.dat, texSpec, 0, 0, otherState.calcDimsStruct.imageTextureWidth, otherState.calcDimsStruct.imageTextureHeight)
                    end
                end
                
                otherState.currentlyDispDat = singleSlDat
                otherState.currentDisplayedSlice = newSlice
                otherState.isSliceChanged = true
            end
        end
        changed = true
    end

    t_total_ms = (time_ns() - t_total) / 1e6
    @info "[BENCH] Next Lesion (Fast Right-Click Emulation): $(round(t_total_ms, digits=1))ms"
    return changed
end

# TP navigation state: compact cache holding only base axial volumes
"""Compact cache entry storing only base (axial) volumes at minimal precision."""
mutable struct TpCacheEntry
    ct::Array{Float32,3}
    pet::Array{Float32,3}
    mask::Union{Array{Int8,3}, Array{Int16,3}}
    bone_surf::BitArray{3}
    bone_marr::BitArray{3}
    anatomy::Union{Nothing, Array{UInt16,3}}  # max_anatomy atlas per-TP (UInt16, 163MB)
    mask_f32::Array{Float32,3}                 # pre-converted Float32 mask (avoids 400ms per switch)
    anat_f32::Array{Float32,3}                 # pre-converted Float32 anatomy (avoids 400ms per switch)
end

const tp_data_cache = Dict{Int, TpCacheEntry}()
const bone_subsegments_cache = Dict{Any, Any}()
const lesion_centroids_cache = Dict{Any, Vector{Int}}()
const last_bone_surf_indices = Dict{Int, Vector{CartesianIndex{3}}}()
const last_bone_marr_indices = Dict{Int, Vector{CartesianIndex{3}}}()
function reactToBoneSubsegResult(data::BoneSubsegResultEvent, stateObjects::Vector{StateDataFields})
    println("reactToBoneSubsegResult: received result for lesion $(data.target_id) on tp $(data.panel_tp)"); flush(stdout)
    bone_subsegments_cache[(data.panel_tp, data.target_id)] = (data.pts_surf, data.pts_marr)
    
    # Re-render if this lesion is still the active one
    if current_active_lesion_id[] == data.target_id && data.target_id > 0
        reactToSyncLesion(SyncLesionEvent(data.target_id), stateObjects)
    end
end

const tp_loader_ref = Ref{Any}(nothing)
const io_channel = Ref{Any}(nothing)
const main_event_channel = Ref{Any}(nothing)
export register_tp_loader!, register_main_channel!, get_or_load_tp_data, last_bone_surf_indices, last_bone_marr_indices
export TpCacheEntry, invalidate_suv_for_lesion

function register_main_channel!(ch::Channel)
    main_event_channel[] = ch
end

# IO channel message types for background TP loading/eviction
struct PreloadTPMessage
    tp_idx::Int
end

struct EvictAndPreloadMessage
    evict_tps::Vector{Int}
    preload_tps::Vector{Int}
end

const io_channel = Ref{Any}(nothing)
const _io_task_started = Ref(false)
export io_channel, PreloadTPMessage, EvictAndPreloadMessage

"""Start the IO consumer task if not already running. Must be called at runtime, not precompile time."""
function _ensure_io_task!()
    _io_task_started[] && return
    _io_task_started[] = true
    io_channel[] = Channel{Any}(16)
    Threads.@spawn begin
        for msg in io_channel[]
            try
                if msg isa PreloadTPMessage
                    tp = msg.tp_idx
                    if !haskey(tp_data_cache, tp) && tp_loader_ref[] !== nothing
                        t = @elapsed begin
                            entry = tp_loader_ref[](tp)
                            entry !== nothing && (tp_data_cache[tp] = entry)
                        end
                        println("  [IO] Preloaded TP $tp in $(round(t, digits=1))s"); flush(stdout)
                    else
                        println("  [IO] TP $tp already cached, skipping"); flush(stdout)
                    end
                elseif msg isa EvictAndPreloadMessage
                    # Evict first to free memory before loading new data
                    for tp in msg.evict_tps
                        delete!(tp_data_cache, tp)
                    end
                    if !isempty(msg.evict_tps)
                        GC.gc(false)
                        println("  [IO] Evicted TPs $(msg.evict_tps)"); flush(stdout)
                    end
                    # Then preload neighbors
                    for tp in msg.preload_tps
                        if !haskey(tp_data_cache, tp) && tp_loader_ref[] !== nothing
                            t = @elapsed begin
                                entry = tp_loader_ref[](tp)
                                entry !== nothing && (tp_data_cache[tp] = entry)
                            end
                            println("  [IO] Preloaded TP $tp in $(round(t, digits=1))s"); flush(stdout)
                        end
                    end
                end
            catch e
                println("  [IO] Error processing message: $e"); flush(stdout)
            end
        end
    end
    println("  [IO] Background IO task started"); flush(stdout)
end

function register_tp_loader!(fn)
    tp_loader_ref[] = fn
    _ensure_io_task!()  # Start IO consumer task on first registration
    
    # Eagerly preload neighbor TPs — skip if already preloaded before GUI launch
    Threads.@spawn begin
        sleep(0.5)  # Allow initial display to finish first
        tp_indices = sort(collect(keys(tp_labels)))
        all_cached = all(tp_idx -> haskey(tp_data_cache, tp_idx), tp_indices)
        if all_cached
            println("  [STARTUP] All TPs already in cache, skipping background preload"); flush(stdout)
        else
            for tp_idx in tp_indices
                if tp_idx != 0 && !haskey(tp_data_cache, tp_idx) && io_channel[] !== nothing
                    try
                        put!(io_channel[], PreloadTPMessage(tp_idx))
                        println("  [STARTUP] Dispatched background preload for TP $tp_idx"); flush(stdout)
                    catch; end
                end
            end
        end
    end
end

function get_or_load_tp_data(idx::Int)
    if haskey(tp_data_cache, idx)
        return tp_data_cache[idx]
    elseif tp_loader_ref[] !== nothing
        entry = tp_loader_ref[](idx)
        if entry !== nothing
            tp_data_cache[idx] = entry
            return entry
        end
    end
    return nothing
end
# Helper to extract existing bone array reference (WITHOUT zeroing — caller decides)
function get_existing_bone_array(stateObject, name)
    for scrDat in stateObject.onScrollData.dataToScroll
        if scrDat.name == name
            arr = scrDat.dat isa PermutedDimsArray ? parent(scrDat.dat) : scrDat.dat
            return arr
        end
    end
    # Fallback — no bone array found in this panel's dataToScroll
    return nothing
end

"""Load a TpCacheEntry into a specific panel, using PermutedDimsArray for sag/cor views.
Pre-converted Float32 arrays can be passed to avoid redundant conversions across panels."""
function _load_tp_from_entry!(stateObjects, entry::TpCacheEntry, panel_idx;
                              mask_f32=nothing, bone_s_f32=nothing, bone_m_f32=nothing)
    if panel_idx > length(stateObjects)
        return
    end
    
    # New data arrays → old bone overlay indices are stale
    delete!(last_bone_surf_indices, panel_idx)
    delete!(last_bone_marr_indices, panel_idx)
    
    # Use pre-computed Float32 arrays from TpCacheEntry (avoids ~800ms per call)
    if mask_f32 === nothing
        mask_f32 = entry.mask_f32
    end
    if bone_s_f32 === nothing
        bone_s_f32 = Float32.(entry.bone_surf)
    end
    if bone_m_f32 === nothing
        bone_m_f32 = Float32.(entry.bone_marr)
    end
    anat_f32 = entry.anat_f32
    
    # Use PermutedDimsArray for zero-copy views on CT/PET/Mask/Anatomy,
    # but bone arrays MUST be independent per-panel: reactToSyncLesion writes
    # bone subseg indices per-panel in panel-specific coordinate systems,
    # so shared arrays get polluted by cross-panel writes (causing doubling).
    
    function get_or_create_bone(panel_idx, name, req_size)
        arr = get_existing_bone_array(stateObjects[panel_idx], name)
        if arr !== nothing && size(arr) == req_size
            fill!(arr, 0.0f0)
            return arr
        end
        return zeros(Float32, req_size)
    end

    panel_voxels = if panel_idx == 3  # Sagittal (Y,Z,X)
        sz = (size(entry.ct, 2), size(entry.ct, 3), size(entry.ct, 1))
        Any[("CT", PermutedDimsArray(entry.ct, (2,3,1))),
            ("PET", PermutedDimsArray(entry.pet, (2,3,1))),
            ("Mask", PermutedDimsArray(mask_f32, (2,3,1))),
            ("Bone_Surface", get_or_create_bone(panel_idx, "Bone_Surface", sz)),
            ("Bone_Marrow", get_or_create_bone(panel_idx, "Bone_Marrow", sz)),
            ("Anatomy", PermutedDimsArray(anat_f32, (2,3,1)))]
    elseif panel_idx == 4  # Coronal (X,Z,Y)
        sz = (size(entry.ct, 1), size(entry.ct, 3), size(entry.ct, 2))
        Any[("CT", PermutedDimsArray(entry.ct, (1,3,2))),
            ("PET", PermutedDimsArray(entry.pet, (1,3,2))),
            ("Mask", PermutedDimsArray(mask_f32, (1,3,2))),
            ("Bone_Surface", get_or_create_bone(panel_idx, "Bone_Surface", sz)),
            ("Bone_Marrow", get_or_create_bone(panel_idx, "Bone_Marrow", sz)),
            ("Anatomy", PermutedDimsArray(anat_f32, (1,3,2)))]
    elseif panel_idx == 2  # PET-only
        Any[("PET", entry.pet)]
    else  # Axial (panels 1, 5) — each gets its own copy
        sz = size(entry.ct)
        Any[("CT", entry.ct), ("PET", entry.pet), ("Mask", mask_f32),
            ("Bone_Surface", get_or_create_bone(panel_idx, "Bone_Surface", sz)),
            ("Bone_Marrow", get_or_create_bone(panel_idx, "Bone_Marrow", sz)),
            ("Anatomy", anat_f32)]
    end
    
    # Insert manualModif at index 2 — reuse existing buffer from stateObject when possible
    existing_manual = nothing
    if !isempty(stateObjects[panel_idx].onScrollData.dataToScroll)
        for scrDat in stateObjects[panel_idx].onScrollData.dataToScroll
            if scrDat.name == "manualModif"
                if size(scrDat.dat) == size(panel_voxels[1][2])
                    existing_manual = scrDat.dat
                    fill!(existing_manual, 0.0f0)
                end
                break
            end
        end
    end
    manual_buf = existing_manual !== nothing ? existing_manual : zeros(Float32, size(panel_voxels[1][2]))
    
    if panel_idx != 2 && (length(panel_voxels) < 2 || panel_voxels[2][1] != "manualModif")
        insert!(panel_voxels, 2, ("manualModif", manual_buf))
    elseif panel_idx == 2
        insert!(panel_voxels, 1, ("manualModif", manual_buf))
    end
    
    newDataToScroll = StructsManag.getThreeDims(panel_voxels)
    stateObjects[panel_idx].onScrollData.dataToScroll = newDataToScroll
    stateObjects[panel_idx].onScrollData.nameIndexes = DataStructs.getLocationDict(newDataToScroll)
    
    # Track which TP this panel holds
    stateObjects[panel_idx].onScrollData.currentTpIndex = current_tp_index[]
    stateObjects[panel_idx].onScrollData.totalTpCount = length(tp_labels)
    stateObjects[panel_idx].onScrollData.tpIndices = sort(collect(keys(tp_labels)))
    
    dimToScroll = stateObjects[panel_idx].onScrollData.dimensionToScroll
    if !isempty(newDataToScroll)
        stateObjects[panel_idx].onScrollData.slicesNumber = Int32(size(newDataToScroll[1].dat, dimToScroll))
    end
    stateObjects[panel_idx].currentDisplayedSlice = max(1, stateObjects[panel_idx].onScrollData.slicesNumber ÷ 2)
    stateObjects[panel_idx].currentlyDispDat = SingleSliceDat(sliceNumber=0)
end


"""Invalidate cached SUV and centroid data for a lesion after mask modification."""
function invalidate_suv_for_lesion(lesion_id::Int, tp_idx::Int)
    try
        LMW = Main.MedEye3d.LesionMetadataWindow
        if isdefined(LMW, :_lesion_suv_cache)
            delete!(LMW._lesion_suv_cache, (tp_idx, lesion_id))
        end
        if isdefined(LMW, :_db_dirty)
            LMW._db_dirty[] = true
        end
    catch; end
    delete!(lesion_centroids_cache, (tp_idx, lesion_id))
    delete!(lesion_centroids_cache, lesion_id)
    println("  [SUV] Invalidated cache for lesion $lesion_id @ TP $tp_idx"); flush(stdout)
end
const global_bone_atlas = Ref{Any}(nothing)
const global_organ_mapping = Ref{Dict{Int,String}}(Dict{Int,String}())  # lesion_id -> TS organ name (from map_lesions_to_organs)
const current_tp_index = Ref(0)
const tp_labels = Dict{Int, String}()  # tp_index -> display label (e.g. "PET TP0")
const tp_descriptions = Dict{Int, String}() # tp_index -> radiological description (German)
const tp_english_descriptions = Dict{Int, String}() # tp_index -> English radiological description

# PET volume per TP for SUV computation: tp_index -> 3D Float32 array (axial orientation, Y-reversed)
const pet_volumes_cache = Dict{Int, Array{Float32, 3}}()
# TotalSegmentator atlas + names for background SUV reference organs
const global_ts_atlas = Ref{Any}(nothing)          # 3D UInt8/Int array (axial, Y-reversed)
const global_ts_names = Ref{Dict{Int,String}}(Dict{Int,String}())  # TS label -> organ name
# Patient identification
const patient_id = Ref{String}("")
# Path to preprocessed HDF5 file (single source of truth for JSON metadata)
const h5_path_ref = Ref{String}("")
# Study modalities per TP: tp_index -> "PET" or "SPECT"
const tp_modalities = Dict{Int, String}()
# Current PET/CT blend weight (0.0=CT only, 1.0=full PET overlay)
const current_pet_blend = Ref(1.0f0)
# Total axial slices for edge-slice artefact detection
const volume_z_size = Ref(0)
# Per-TP anatomy labels: tp_index → Dict{Int,String} for cursor readout
const anatomy_labels_cache = Dict{Int, Dict{Int,String}}()

export tp_data_cache, bone_subsegments_cache, lesion_centroids_cache, global_bone_atlas, global_organ_mapping, current_tp_index, tp_labels, tp_descriptions, tp_english_descriptions
export compare_mode, compare_right_tp, tp_switched, get_node_name_for_tp, tp_node_names
export pet_volumes_cache, global_ts_atlas, global_ts_names, patient_id, h5_path_ref, tp_modalities, volume_z_size, anatomy_labels_cache


function reactToChangeTimePoint(data::ChangeTimePointEvent, stateObjects::Vector{StateDataFields})
    t_total = time_ns()
    if isempty(tp_labels)
        println("No TP labels loaded. TP navigation disabled."); flush(stdout)
        return
    end
    
    # Get sorted TP indices
    tp_indices = sort(collect(keys(tp_labels)))
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
        t_load = @elapsed begin
            entry_left = get_or_load_tp_data(new_tp)
        end
        if DEBUG_VERBOSE[]; println("  [BENCH] get_or_load_tp_data(left): $(round(t_load, digits=3))s"); flush(stdout); end
        
        t_panel_left = @elapsed begin
            if entry_left !== nothing
                # Extract existing display arrays to avoid 1.3 GB reallocation
                lm = entry_left.mask_f32
                ls = get_existing_bone_array(stateObjects[1], "Bone_Surface")
                lmr = get_existing_bone_array(stateObjects[1], "Bone_Marrow")
                _load_tp_from_entry!(stateObjects, entry_left, 1; mask_f32=lm, bone_s_f32=ls, bone_m_f32=lmr)
            end
        end
        if DEBUG_VERBOSE[]; println("  [BENCH] _load_tp_from_entry!(left): $(round(t_panel_left*1000, digits=1))ms"); flush(stdout); end
        
        # Right panel: next TP chronologically
        next_pos = mod1(new_pos + 1, num_tps)
        right_tp = tp_indices[next_pos]
        compare_right_tp[] = right_tp
        
        t_load_r = @elapsed begin
            entry_right = get_or_load_tp_data(right_tp)
        end
        if DEBUG_VERBOSE[]; println("  [BENCH] get_or_load_tp_data(right): $(round(t_load_r, digits=3))s"); flush(stdout); end
        
        t_panel_right = @elapsed begin
            if entry_right !== nothing
                # Extract existing display arrays to avoid 1.3 GB reallocation
                rm = entry_right.mask_f32
                rs = get_existing_bone_array(stateObjects[5], "Bone_Surface")
                rmr = get_existing_bone_array(stateObjects[5], "Bone_Marrow")
                _load_tp_from_entry!(stateObjects, entry_right, 5; mask_f32=rm, bone_s_f32=rs, bone_m_f32=rmr)
            end
        end
        if DEBUG_VERBOSE[]; println("  [BENCH] _load_tp_from_entry!(right): $(round(t_panel_right*1000, digits=1))ms"); flush(stdout); end
        
        # Skip initial per-panel reactToScroll — reactToSyncLesion below covers [1, 5]
        if DEBUG_VERBOSE[]; println("  [BENCH] skipping redundant initial scroll in compare mode"); flush(stdout); end
        
        # Re-apply bone overlay for active lesion after TP data replacement
        if current_active_lesion_id[] > 0
            reactToSyncLesion(SyncLesionEvent(current_active_lesion_id[]), stateObjects)
        end
        
        right_label = get(tp_labels, right_tp, "TP $right_tp")
        println("Compare: Left=$label, Right=$right_label"); flush(stdout)
    else
        # Normal mode: load current TP into all panels
        t_load = @elapsed begin
        entry = get_or_load_tp_data(new_tp)
        end
        if DEBUG_VERBOSE[]; println("  [BENCH] get_or_load_tp_data: $(round(t_load*1000, digits=1))ms (cached=$(haskey(tp_data_cache, new_tp)))"); flush(stdout); end
        
        if entry !== nothing
            # Pre-convert arrays ONCE for all panels
            t_convert = @elapsed begin
                mask_f32 = entry.mask_f32
            end
            if DEBUG_VERBOSE[]; println("  [BENCH] Float32(mask): $(round(t_convert*1000, digits=1))ms ($(size(entry.mask)) $(eltype(entry.mask)))"); flush(stdout); end
            
            t_bone_convert = @elapsed begin
                bone_s_f32 = get_existing_bone_array(stateObjects[1], "Bone_Surface")
                bone_m_f32 = get_existing_bone_array(stateObjects[1], "Bone_Marrow")
            end
            if DEBUG_VERBOSE[]; println("  [BENCH] Extract existing bone_surf+marr arrays: $(round(t_bone_convert*1000, digits=1))ms"); flush(stdout); end
            
            t_panels = @elapsed begin
                for i in [1, 2, 3, 4]
                    if i <= length(stateObjects)
                        _load_tp_from_entry!(stateObjects, entry, i;
                            mask_f32=mask_f32, bone_s_f32=bone_s_f32, bone_m_f32=bone_m_f32)
                    end
                end
                if length(stateObjects) >= 5
                    _load_tp_from_entry!(stateObjects, entry, 5;
                        mask_f32=mask_f32, bone_s_f32=bone_s_f32, bone_m_f32=bone_m_f32)
                end
            end
            # Skip initial per-panel reactToScroll here — reactToSyncLesion below
            # will do its own reactToScroll for all active panels, which uploads textures.
            # Only do a minimal scroll for panels NOT covered by reactToSyncLesion.
            # reactToSyncLesion covers active_panel_indices = [1,2,3,4] in normal mode.
            # Panel 5 is only used in compare mode (handled separately above).
            if DEBUG_VERBOSE[]; println("  [BENCH] _load_tp_from_entry! completed, skipping redundant initial scroll"); flush(stdout); end
            
            # Re-apply bone overlay + navigate to active lesion (or lesion 1 if none)
            empty!(last_bone_surf_indices)
            empty!(last_bone_marr_indices)
            t_bone_overlay = @elapsed begin
                lid = current_active_lesion_id[] > 0 ? current_active_lesion_id[] : 1
                try
                    reactToSyncLesion(SyncLesionEvent(lid), stateObjects)
                    println("Synced to Lesion $lid for $label"); flush(stdout)
                catch e
                    println("WARNING: Failed to sync Lesion $lid on TP change: $e"); flush(stdout)
                end
            end
            if DEBUG_VERBOSE[]; println("  [BENCH] bone overlay (reactToSyncLesion): $(round(t_bone_overlay*1000, digits=1))ms"); flush(stdout); end
        end
    end
    # All TPs pre-loaded before GUI launch — no eviction or preload dispatch needed
    
    # SUV precompute + CT Docker preload: fire-and-forget in background (non-blocking)
    let tp_for_bg = new_tp, label_for_bg = label
        Threads.@spawn begin
            try
                LMW = Main.MedEye3d.LesionMetadataWindow
                if haskey(tp_data_cache, tp_for_bg)
                    cached_entry = tp_data_cache[tp_for_bg]
                    unique_ids = Set{Int}()
                    for v in cached_entry.mask
                        iv = Int(v)
                        iv > 0 && push!(unique_ids, iv)
                    end
                    for lid in unique_ids
                        key = (tp_for_bg, lid)
                        !haskey(LMW._lesion_suv_cache, key) && 
                            try LMW._lesion_suv_cache[key] = LMW.compute_lesion_suv_string(lid, tp_for_bg) catch; end
                    end
                    println("  [BG] SUV precomputed for $(length(unique_ids)) lesions"); flush(stdout)
                end
            catch; end
            
            try
                if haskey(tp_data_cache, tp_for_bg)
                    InferenceClient.preload_ct_for_nninteractive(Array{Float32,3}(tp_data_cache[tp_for_bg].ct))
                    println("[BG] CT preload initiated for $label_for_bg"); flush(stdout)
                end
            catch; end
        end
    end
    
    t_total_ms = (time_ns() - t_total) / 1e6
    @info "[BENCH] Next/Prev TP Total: $(round(t_total_ms, digits=1))ms"
    tp_switched[] = tp_switched[] + 1
end

function reactToToggleLesion(data::ToggleLesionEvent, stateObjects::Vector{StateDataFields})
    for stateObject in stateObjects
        for textSpec in stateObject.mainForDisplayObjects.listOfTextSpecifications
            if textSpec.isMultiDiscreteMask
                textSpec.isVisible = !textSpec.isVisible
                
                # UBO update happens in consumer loop
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
    # (Removed synchronous computation - it is now delegated to the async _get_or_compute_bone_subseg below)

    # Invalidate cache for the active lesion on all visible TPs so it recomputes after AI paints it
    target_lid = data.active_id
    delete!(bone_subsegments_cache, (current_tp_index[], target_lid))
    delete!(bone_subsegments_cache, (get_node_name_for_tp(current_tp_index[]), target_lid))
    
    panel5_lesion_id = target_lid
    if compare_mode[] && length(stateObjects) >= 5 && target_lid > 0
        try
            left_node = get_node_name_for_tp(current_tp_index[])
            right_node = get_node_name_for_tp(compare_right_tp[])
            match_mod = isdefined(Main, :MedEye3d) && isdefined(Main.MedEye3d, :LesionAssociation) ? Main.MedEye3d.LesionAssociation : nothing
            if match_mod !== nothing
                matched_ids = match_mod.find_cross_tp_lesion(left_node, target_lid, right_node)
                if !isempty(matched_ids)
                    panel5_lesion_id = matched_ids[1]
                end
            end
        catch e
        end
        delete!(bone_subsegments_cache, (compare_right_tp[], panel5_lesion_id))
        delete!(bone_subsegments_cache, (get_node_name_for_tp(compare_right_tp[]), panel5_lesion_id))
    end

    # Update bone surface & marrow textures in all panels
    for (panel_idx, stateObject) in enumerate(stateObjects)
        panel_tp = (panel_idx == 5 && compare_mode[]) ? compare_right_tp[] : current_tp_index[]
        panel_lid = (panel_idx == 5 && compare_mode[]) ? panel5_lesion_id : target_lid
        
        panel_surf_pts, panel_marr_pts = try
            _get_or_compute_bone_subseg(stateObject, panel_lid, panel_tp)
        catch e
            println("Failed to recalc bone for panel $panel_idx in reactToAIInferenceResult: $e")
            (CartesianIndex{3}[], CartesianIndex{3}[])
        end
        surf_indices = if panel_idx == 3
            [CartesianIndex(I[2], I[3], I[1]) for I in panel_surf_pts]
        elseif panel_idx == 4
            [CartesianIndex(I[1], I[3], I[2]) for I in panel_surf_pts]
        else
            panel_surf_pts
        end
        marr_indices = if panel_idx == 3
            [CartesianIndex(I[2], I[3], I[1]) for I in panel_marr_pts]
        elseif panel_idx == 4
            [CartesianIndex(I[1], I[3], I[2]) for I in panel_marr_pts]
        else
            panel_marr_pts
        end
        
        for scrDat in stateObject.onScrollData.dataToScroll
            if scrDat.name == "Bone_Surface"
                if haskey(last_bone_surf_indices, panel_idx) && !isempty(last_bone_surf_indices[panel_idx])
                    scrDat.dat[last_bone_surf_indices[panel_idx]] .= 0.0f0
                end
                if !isempty(surf_indices)
                    scrDat.dat[surf_indices] .= 1.0f0
                end
                last_bone_surf_indices[panel_idx] = surf_indices
            elseif scrDat.name == "Bone_Marrow"
                if haskey(last_bone_marr_indices, panel_idx) && !isempty(last_bone_marr_indices[panel_idx])
                    scrDat.dat[last_bone_marr_indices[panel_idx]] .= 0.0f0
                end
                if !isempty(marr_indices)
                    scrDat.dat[marr_indices] .= 1.0f0
                end
                last_bone_marr_indices[panel_idx] = marr_indices
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
        entry = tp_data_cache[tp_idx]
        if entry.mask isa Array{Int8, 3}
            entry.mask .= round.(Int8, seg_vol)
        else
            entry.mask .= round.(Int16, seg_vol)
        end
    end

    # Ensure mask uniform displays the active lesion
    for stateObject in stateObjects
        for textSpec in stateObject.mainForDisplayObjects.listOfTextSpecifications
            if textSpec.name == "Mask" || textSpec.name == "segmentation"
                textSpec.minAndMaxValue = Float32.([data.active_id, data.active_id])
            elseif textSpec.name == "manualModif"
                textSpec.minAndMaxValue = Float32.([0.0, 1000.0])
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
    println("Bone subsegmentation (manual) triggered for lesion $(data.lesion_id)"); flush(stdout)
    
    tp_idx = current_tp_index[]
    
    # Invalidate bone subsegment cache for this lesion
    delete!(bone_subsegments_cache, (tp_idx, data.lesion_id))
    delete!(bone_subsegments_cache, (get_node_name_for_tp(tp_idx), data.lesion_id))
    delete!(bone_subsegments_cache, data.lesion_id)
    if compare_mode[]
        panel5_lesion_id = data.lesion_id
        if length(stateObjects) >= 5 && data.lesion_id > 0
            try
                left_node = get_node_name_for_tp(tp_idx)
                right_node = get_node_name_for_tp(compare_right_tp[])
                match_mod = isdefined(Main, :MedEye3d) && isdefined(Main.MedEye3d, :LesionAssociation) ? Main.MedEye3d.LesionAssociation : nothing
                if match_mod !== nothing
                    matched_ids = match_mod.find_cross_tp_lesion(left_node, data.lesion_id, right_node)
                    if !isempty(matched_ids)
                        panel5_lesion_id = matched_ids[1]
                    end
                end
            catch e
            end
        end
        delete!(bone_subsegments_cache, (compare_right_tp[], panel5_lesion_id))
        delete!(bone_subsegments_cache, (get_node_name_for_tp(compare_right_tp[]), panel5_lesion_id))
    end
    
    # Invalidate centroid cache as well
    delete!(lesion_centroids_cache, (tp_idx, data.lesion_id))
    delete!(lesion_centroids_cache, data.lesion_id)
    
    # Trigger an async recomputation by forcing a sync lesion update
    reactToSyncLesion(SyncLesionEvent(data.lesion_id), stateObjects)
end

function reactToMapLink(data::MapLinkEvent, stateObjects::Vector{StateDataFields})
    println("Map Link triggered. Linking lesions: src=$(data.src_ids) to dst=$(data.dst_ids)"); flush(stdout)
    if length(stateObjects) > 1
        # LesionAssociation.map_link("TP1", "TP2", data.src_ids, data.dst_ids)
        println("Successfully mapped between TP1 and TP2"); flush(stdout)
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
        for textSpec in stateObject.mainForDisplayObjects.listOfTextSpecifications
            if textSpec.name == "Bone_Mask" || textSpec.name == "bone_mask" || textSpec.name == "bone" || textSpec.name == "Organ_Mask" || textSpec.name == "organ_mask" || textSpec.name == "Bone_Surface" || textSpec.name == "Bone_Marrow"
                textSpec.isVisible = data.active
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
    elseif data.layer == 4
        "Anatomy"
    else
        ""
    end
    
    toggled_count = 0
    for (si, state) in enumerate(stateObjects)
        for textSpec in state.mainForDisplayObjects.listOfTextSpecifications
            if (tex_target == "Mask" && (textSpec.name == "Mask" || textSpec.name == "manualModif" || textSpec.name == "segmentation")) ||
               (textSpec.name == tex_target)
                textSpec.isVisible = data.active
                toggled_count += 1
                println("  Panel $si: set isVisible=$(data.active) for texture '$(textSpec.name)'")
            end
        end
    end
    println("  Toggled $toggled_count textures total for layer=$(data.layer)")
    flush(stdout)
    
    # If bone surface or marrow is toggled, also sync dataToScroll buffer directly
    if data.layer == 2 || data.layer == 3
        cur_lid = (current_active_lesion_id[] > 0) ? current_active_lesion_id[] : round(Int, stateObjects[1].valueForMasToSet.value)
        println("  Bone data sync: cur_lid=$cur_lid, in_cache=$(haskey(bone_subsegments_cache, (current_tp_index[], cur_lid)))")
        flush(stdout)
        for (panel_idx, stateObject) in enumerate(stateObjects)
            for scrDat in stateObject.onScrollData.dataToScroll
                if (data.layer == 2 && scrDat.name == "Bone_Surface") || (data.layer == 3 && scrDat.name == "Bone_Marrow")
                    if !data.active
                        if data.layer == 2 && haskey(last_bone_surf_indices, panel_idx)
                            scrDat.dat[last_bone_surf_indices[panel_idx]] .= 0.0f0
                            delete!(last_bone_surf_indices, panel_idx)
                        elseif data.layer == 3 && haskey(last_bone_marr_indices, panel_idx)
                            scrDat.dat[last_bone_marr_indices[panel_idx]] .= 0.0f0
                            delete!(last_bone_marr_indices, panel_idx)
                        else
                            fill!(scrDat.dat, 0.0f0)
                        end
                    elseif cur_lid > 0
                        panel_tp = (panel_idx == 5 && compare_mode[]) ? compare_right_tp[] : current_tp_index[]
                        panel_lid = cur_lid
                        if panel_idx == 5 && compare_mode[] && length(stateObjects) >= 5 && cur_lid > 0
                            try
                                left_node = get_node_name_for_tp(current_tp_index[])
                                right_node = get_node_name_for_tp(compare_right_tp[])
                                match_mod = isdefined(Main, :MedEye3d) && isdefined(Main.MedEye3d, :LesionAssociation) ? Main.MedEye3d.LesionAssociation : nothing
                                if match_mod !== nothing
                                    matched_ids = match_mod.find_cross_tp_lesion(left_node, cur_lid, right_node)
                                    if !isempty(matched_ids)
                                        panel_lid = matched_ids[1]
                                    end
                                end
                            catch e
                            end
                        end
                        panel_surf_pts, panel_marr_pts = try
                            _get_or_compute_bone_subseg(stateObject, panel_lid, panel_tp)
                        catch e
                            println("Failed to recalc bone for panel $panel_idx (toggle): $e")
                            (CartesianIndex{3}[], CartesianIndex{3}[])
                        end
                        panel_pts = (data.layer == 2) ? panel_surf_pts : panel_marr_pts
                        # Use canonical indices matching reactToSyncLesion and reactToActiveLesionChanged
                        indices = if panel_idx == 3 # Sagittal (Y, Z, X)
                            [CartesianIndex(I[2], I[3], I[1]) for I in panel_pts]
                        elseif panel_idx == 4 # Coronal (X, Z, Y)
                            [CartesianIndex(I[1], I[3], I[2]) for I in panel_pts]
                        else # Axial (X, Y, Z)
                            panel_pts
                        end
                        if data.layer == 2
                            if haskey(last_bone_surf_indices, panel_idx) && !isempty(last_bone_surf_indices[panel_idx])
                                scrDat.dat[last_bone_surf_indices[panel_idx]] .= 0.0f0
                            end
                            if !isempty(indices)
                                scrDat.dat[indices] .= 1.0f0
                            end
                            last_bone_surf_indices[panel_idx] = indices
                        elseif data.layer == 3
                            if haskey(last_bone_marr_indices, panel_idx) && !isempty(last_bone_marr_indices[panel_idx])
                                scrDat.dat[last_bone_marr_indices[panel_idx]] .= 0.0f0
                            end
                            if !isempty(indices)
                                scrDat.dat[indices] .= 1.0f0
                            end
                            last_bone_marr_indices[panel_idx] = indices
                        end
                    end
                end
            end
        end
    end
    
    # Re-render all visible panels in one batch
    visible_panels = Int[]
    for idx in 1:length(stateObjects)
        if sum(abs.(stateObjects[idx].calcDimsStruct.mainImageQuadVert)) > 0.01f0
            push!(visible_panels, idx)
        end
    end
    if !isempty(visible_panels)
        try
            ReactToScroll.reactToScrollMultiPanel!(visible_panels, stateObjects)
        catch e
            println("reactToScrollMultiPanel! ERROR during visibility toggle: $e")
            println(sprint(showerror, e, catch_backtrace()))
            flush(stdout)
        end
    end
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
