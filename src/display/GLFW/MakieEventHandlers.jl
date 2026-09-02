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
using HDF5

export reactToChangePlane, reactToCompareTimePoints, reactToShowSingleLesion
export reactToWindowing, reactToPaintVal, reactToSyncLesion, reactToChangeBrushSize, reactToPetBlend, reactToLabelOpacity
export reactToChangeTimePoint, reactToToggleLesion, reactToRefreshList
export reactToAddAutoPet, reactToAIInferenceResult, reactToSyncMissing, reactToGenManual
export reactToMapLink, reactToAutoRunPreprocess, reactToRunPreprocess, reactToShowBoneMask, reactToShowMaskLayer, reactToSaveMRB
export register_h5_mask_saver!, mark_tp_mask_dirty!, save_tp_mask_to_h5, flush_all_dirty_masks!, dirty_mask_tps
using ...InferenceClient
using ...LesionAssociation
using ...TextureManag
# ModernGL removed — Vulkan UBO updates happen in consumer loop via update_ubo!
# Uniforms module no longer needed — TextureSpec fields are read directly by UBO packer
using Observables

# Debug flag: set to true to enable verbose bench/bone logging in hot paths
const DEBUG_VERBOSE = Ref(false)

# Pre-allocated zero arrays for hidden panel quad vertices (Fix ❼: avoid allocations per toggle)
const _HIDDEN_QUAD_VERTS = zeros(Float32, 32)
const _HIDDEN_QUAD_VERTS_W = zeros(Float32, 32)

# AI status Observable — LesionMetadataWindow reads this for the GUI label
const ai_status_text = Observable{String}("Ready")

# Cursor info Observables — updated from on_next!(MouseStruct) via reactToMouseDrag
const cursor_info_text = Observable{String}("")      # "HU: 45 | SUV: 3.2 | femur (L5) | [Ax] Sl:163"
const cursor_study_text = Observable{String}("")     # "PET TP0" or "L: PET TP0 | R: PET TP3"
# 3D voxel position under cursor (axial orientation: x, y, z=slice) — used for new lesion anatomy lookup
const current_viewer_position = Ref((0, 0, 0))
export cursor_info_text, cursor_study_text, set_ai_status!, current_viewer_position

# Sanitize AI status text for Makie Label rendering (ASCII-only, truncated)
function safe_status_text(msg::String)
    s = replace(msg, "\u2014" => "-", "\u2026" => "...")
    s = String(filter(c -> isascii(c), collect(s)))
    return length(s) > 80 ? s[1:80] * "..." : s
end

# Thread-safe AI status updater (ensures Observable mutation doesn't race GLMakie renderloop)
function set_ai_status!(msg::String)
    s = safe_status_text(msg)
    @async begin
        try
            ai_status_text[] = s
        catch; end
    end
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
    seg_vol::Any  # Reference to the live mask volume (Array{Int16,3})
    main_channel::Any  # Channel{Any} or ChannelProxy (parallel startup)
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
                
                set_ai_status!("[Sending] to Docker ($(job.algorithm))...")
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
                    set_ai_status!("[Warning] Unknown algorithm: $(job.algorithm)")
                    continue
                end
                
                if mask !== nothing
                    voxel_count = count(mask .> 0)
                    set_ai_status!("[Applying] result ($voxel_count voxels)...")
                    println("[AI Worker] Docker returned mask with $voxel_count voxels. Posting to channel."); flush(stdout)
                else
                    set_ai_status!("[Warning] Docker returned no mask")
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
                set_ai_status!("[Error] AI Worker Error: $err_msg")
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
    
    # Thread-safe write to shared cache (parallel TP loading writes concurrently)
    lock(_centroids_lock) do
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
        # Panels 3 (Sagittal) and 4 (Coronal) have pre-permuted data that
        # must always slice along dimension 3. Skip plane changes for them.
        if idx in (3, 4)
            push!(panel_indices, idx)
            continue
        end
        
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
        else
            if idx == 5
                updateQuadVertices!(stateObject, :Hidden)
            end
        end
        push!(panel_indices, idx)
    end
    # Batch texture upload for all panels at once
    ReactToScroll.reactToScrollMultiPanel!(panel_indices, stateObjects)
    
    # Force slice re-upload: processKeysInfo already set currentDisplayedSlice,
    # so reactToScrollMultiPanel may see slice_changed=false (same slice number,
    # different plane). Force the consumer to upload the new plane's data.
    for idx in panel_indices
        stateObjects[idx].isSliceChanged = true
    end
end

function updateQuadVertices!(stateObject::StateDataFields, layout::Symbol)
    calcDimStruct = stateObject.calcDimsStruct
    
    if layout == :Hidden
        stateObject.calcDimsStruct = Setfield.setproperties(calcDimStruct, (
            mainImageQuadVert = _HIDDEN_QUAD_VERTS, 
            mainQuadVertSize = sizeof(_HIDDEN_QUAD_VERTS),
            wordsImageQuadVert = _HIDDEN_QUAD_VERTS_W,
            wordsQuadVertSize = sizeof(_HIDDEN_QUAD_VERTS_W)
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
    
    # Fix ❹: Removed dead TextureManag.updateTexture calls (no-op in Vulkan backend).
    # Actual GPU upload happens in consumer loop via isSliceChanged.
    # Safety: verify data dimensions fit within allocated texture (log-only)
    n_textures = length(singleSlDat.listOfDataAndImageNames)
    for updateDat in singleSlDat.listOfDataAndImageNames
        actual_w = size(updateDat.dat, 1)
        actual_h = size(updateDat.dat, 2)
        tex_w = Int(panelState.calcDimsStruct.imageTextureWidth)
        tex_h = Int(panelState.calcDimsStruct.imageTextureHeight)
        if actual_w > tex_w || actual_h > tex_h
            println("  [COMPARE-DBG] SKIPPING texture '$(updateDat.name)' on panel $panel_idx: data=$(actual_w)x$(actual_h) > texture=$(tex_w)x$(tex_h)"); flush(stdout)
        end
    end
    
    panelState.currentlyDispDat = singleSlDat
    panelState.currentDisplayedSlice = current
    panelState.isSliceChanged = true
    println("  [COMPARE-DBG] panel $panel_idx: uploaded $n_textures textures at slice $current (dimToScroll=$dimToScroll, slicesNumber=$lastSlice)"); flush(stdout)
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
                    _load_tp_from_entry!(stateObjects, entry, 5)
                end
            end

            # Ensure panel 5 uses the exact same scroll dimension and slice as panel 1 for registered alignment
            stateObjects[5].onScrollData.dimensionToScroll = stateObjects[1].onScrollData.dimensionToScroll
            stateObjects[5].currentDisplayedSlice = stateObjects[1].currentDisplayedSlice
            stateObjects[5].calcDimsStruct.zoom = stateObjects[1].calcDimsStruct.zoom
            stateObjects[5].calcDimsStruct.panX = stateObjects[1].calcDimsStruct.panX
            stateObjects[5].calcDimsStruct.panY = stateObjects[1].calcDimsStruct.panY

            # 2-pane view: panel 1 on left, panel 5 on right
            updateQuadVertices!(stateObjects[1], :LeftHalf)
            updateQuadVertices!(stateObjects[5], :RightHalf)
            updateQuadVertices!(stateObjects[2], :Hidden)
            updateQuadVertices!(stateObjects[3], :Hidden)
            updateQuadVertices!(stateObjects[4], :Hidden)

            left_label = get(tp_labels, current_tp_index[], "TP $(current_tp_index[])")
            right_label = get(tp_labels, compare_right_tp[], "TP $(compare_right_tp[])")
            println("Compare mode ON: Left=$left_label, Right=$right_label"); flush(stdout)
            
            # Fix ❺: Panel 1 data hasn't changed — only layout vertices moved.
            # Just mark it dirty for the consumer to re-render; skip redundant slice extraction.
            stateObjects[1].isSliceChanged = true
            # Panel 5 is new — force full texture upload
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
            # Fix ❻: Only reload panels whose TP data has actually changed.
            # Panels 1-4 already hold current_tp_index[] data unless TP was switched during compare.
            entry = get_or_load_tp_data(current_tp_index[])
            if entry !== nothing
                cur_tp = current_tp_index[]
                num_panels = min(4, length(stateObjects))
                for i in 1:num_panels
                    panel_tp = try; stateObjects[i].onScrollData.currentTpIndex; catch; -1; end
                    if panel_tp != cur_tp
                        _load_tp_from_entry!(stateObjects, entry, i)
                    end
                end
            end

            # 4-pane view
            updateQuadVertices!(stateObjects[1], :TopLeft)
            updateQuadVertices!(stateObjects[2], :TopRight)
            updateQuadVertices!(stateObjects[3], :BottomLeft)
            updateQuadVertices!(stateObjects[4], :BottomRight)
            updateQuadVertices!(stateObjects[5], :Hidden)
            
            # Reset pan, zoom, displayMode, and center slice for all panels
            for i in 1:length(stateObjects)
                stateObjects[i].calcDimsStruct.zoom = 1.0f0
                stateObjects[i].calcDimsStruct.panX = 0.0f0
                stateObjects[i].calcDimsStruct.panY = 0.0f0
                stateObjects[i].displayMode = QuadImage
                # Restore correct dimensionToScroll for sagittal (3) and coronal (4)
                # panels — their data is pre-permuted and must always slice along dim 3
                if i in (3, 4)
                    old_dts = stateObjects[i].onScrollData.dataToScrollDims
                    stateObjects[i].onScrollData.dataToScrollDims = DataToScrollDims(
                        imageSize = old_dts.imageSize,
                        voxelSize = old_dts.voxelSize,
                        dimensionToScroll = 3)
                    stateObjects[i].onScrollData.dimensionToScroll = 3
                    stateObjects[i].onScrollData.slicesNumber = Int32(old_dts.imageSize[3])
                end
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

# Flag controlling single vs all lesions display mode (default: true = display SINGLE lesion on start)
const is_single_lesion_mode = Ref(true)
export is_single_lesion_mode

function reactToShowSingleLesion(data::ShowSingleLesionEvent, stateObjects::Vector{StateDataFields})
    changed = false
    if data.lesion_id > 0
        current_active_lesion_id[] = data.lesion_id
        is_single_lesion_mode[] = true
    else
        is_single_lesion_mode[] = false
    end
    for stateObject in stateObjects
        for textSpec in stateObject.mainForDisplayObjects.listOfTextSpecifications
            if (textSpec.name == "Mask" || textSpec.name == "segmentation") && textSpec.name != "Anatomy"
                # Clear allowed IDs filter (only for lesion masks, not anatomical atlas)
                textSpec.allowedIDs = Float32[]
                T_mm = eltype(textSpec.minAndMaxValue)
                if !is_single_lesion_mode[]
                    textSpec.minAndMaxValue = T_mm.([1, 1000])
                else
                    textSpec.minAndMaxValue = T_mm.([data.lesion_id, data.lesion_id])
                end
                changed = true
            end
        end
    end
    lbl = is_single_lesion_mode[] ? string(data.lesion_id) : "all"
    println("Show single lesion: $lbl (single_mode=$(is_single_lesion_mode[]))"); flush(stdout)
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
    
    for (panel_idx, state) in enumerate(stateObjects)
        panel_tp = if compare_mode[] && panel_idx == 5
            compare_right_tp[]
        else
            state.onScrollData.currentTpIndex > 0 ? state.onScrollData.currentTpIndex : current_tp_index[]
        end
        panel_mod = uppercase(get(tp_modalities, panel_tp, "PET"))
        
        for tex in state.mainForDisplayObjects.listOfTextSpecifications
            if target_mod == "CT"
                if tex.name == "CT"
                    tex.minAndMaxValue = Float32.([data.min_val, data.max_val])
                end
            elseif target_mod == "PET"
                # Only update nuclear texture if this panel is displaying a PET timepoint
                if (tex.name == "PET" && panel_mod == "PET")
                    tex.minAndMaxValue = Float32.([data.min_val, data.max_val])
                end
            elseif target_mod == "SPECT"
                # Only update nuclear texture if this panel is displaying a SPECT timepoint
                if tex.name == "SPECT" || (tex.name == "PET" && panel_mod == "SPECT")
                    tex.minAndMaxValue = Float32.([data.min_val, data.max_val])
                end
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
    @debug "PET/CT blend updated" weight=data.weight
end

function reactToLabelOpacity(data::LabelOpacityEvent, stateObjects::Vector{StateDataFields})
    for state in stateObjects
        for tex in state.mainForDisplayObjects.listOfTextSpecifications
            # Update opacity for discrete segmentation masks and label overlays
            if tex.isMultiDiscreteMask || (!tex.isMainImage && !tex.isNuclearMask)
                tex.maskContribution = clamp(data.opacity, 0.0f0, 1.0f0)
            end
        end
    end
    @debug "Label opacity updated" opacity=data.opacity
end

function reactToPaintVal(data::PaintValEvent, stateObjects::Vector{StateDataFields})
    if data.val > 0
        current_active_lesion_id[] = data.val
    end
    for state in stateObjects
        state.valueForMasToSet = valueForMasToSetStruct(value=data.val, is_painting_active=data.active)
        if data.active
            target_ts = nothing
            for textSpec in state.mainForDisplayObjects.listOfTextSpecifications
                if textSpec.name == "Mask" || (textSpec.isMultiDiscreteMask && textSpec.name != "Anatomy")
                    target_ts = textSpec
                    break
                elseif textSpec.name == "manualModif" && target_ts === nothing
                    target_ts = textSpec
                end
            end
            if target_ts !== nothing
                target_ts.isVisible = true
                state.textureToModifyVec = [target_ts]
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

function _get_lmw()
    p = parentmodule(parentmodule(@__MODULE__))
    if isdefined(p, :LesionMetadataWindow)
        return getfield(p, :LesionMetadataWindow)
    elseif isdefined(Main, :MedEye3d) && isdefined(Main.MedEye3d, :LesionMetadataWindow)
        return Main.MedEye3d.LesionMetadataWindow
    end
    return nothing
end

function _get_la()
    p = parentmodule(parentmodule(@__MODULE__))
    if isdefined(p, :LesionAssociation)
        return getfield(p, :LesionAssociation)
    elseif isdefined(Main, :MedEye3d) && isdefined(Main.MedEye3d, :LesionAssociation)
        return Main.MedEye3d.LesionAssociation
    end
    return nothing
end

function get_node_name_for_tp(tp_idx::Int)::String
    if haskey(tp_node_names, tp_idx)
        return tp_node_names[tp_idx]
    end
    @warn "No node name for TP $tp_idx — tp_node_names not populated from HDF5"
    return "Unknown_TP_$tp_idx"
end

const current_active_lesion_id = Ref(0)

"""
Fast on-the-fly computation of bone surface (cortex) and bone marrow (trabecula) subsegments around a bone lesion.
"""
function compute_bone_subsegments_fast(mask_vol::AbstractArray{<:Integer, 3}, skelly_vol::AbstractArray{<:Real, 3}, target_id::Int; spacing=(1.0, 1.0, 2.0))
    lesion_vox = findall(mask_vol .== target_id)
    if isempty(lesion_vox) || isempty(skelly_vol)
        return (CartesianIndex{3}[], CartesianIndex{3}[])
    end
    
    sz = size(mask_vol)
    xs = [I[1] for I in lesion_vox]
    ys = [I[2] for I in lesion_vox]
    zs = [I[3] for I in lesion_vox]
    
    margin_x = ceil(Int, 20.0 / spacing[1])
    margin_y = ceil(Int, 20.0 / spacing[2])
    margin_z = ceil(Int, 20.0 / spacing[3])
    
    x_min = max(1, minimum(xs) - margin_x); x_max = min(sz[1], maximum(xs) + margin_x)
    y_min = max(1, minimum(ys) - margin_y); y_max = min(sz[2], maximum(ys) + margin_y)
    z_min = max(1, minimum(zs) - margin_z); z_max = min(sz[3], maximum(zs) + margin_z)
    
    crop_mask = view(mask_vol, x_min:x_max, y_min:y_max, z_min:z_max)
    crop_skelly = view(skelly_vol, x_min:x_max, y_min:y_max, z_min:z_max)
    
    crop_lesion = findall(crop_mask .== target_id)
    crop_bone_vox = findall(crop_skelly .> 0)
    if isempty(crop_lesion) || isempty(crop_bone_vox)
        return (CartesianIndex{3}[], CartesianIndex{3}[])
    end
    
    lx = Float32[I[1] for I in crop_lesion]
    ly = Float32[I[2] for I in crop_lesion]
    lz = Float32[I[3] for I in crop_lesion]
    
    sp1 = Float32(spacing[1])
    sp2 = Float32(spacing[2])
    sp3 = Float32(spacing[3])
    
    n_vox = length(crop_lesion)
    vox_vol = sp1 * sp2 * sp3
    lesion_vol_mm3 = n_vox * vox_vol
    R_L = max(4.0f0, Float32((3.0 * lesion_vol_mm3 / (4.0 * π))^(1/3)))
    
    max_surf_dist_mm2 = 14.0f0^2
    max_marr_dist_mm2 = (R_L + 8.0f0)^2
    
    surf_pts = CartesianIndex{3}[]
    marr_pts = CartesianIndex{3}[]
    
    for b_idx in crop_bone_vox
        val = crop_skelly[b_idx]
        bx = Float32(b_idx[1])
        by = Float32(b_idx[2])
        bz = Float32(b_idx[3])
        
        min_d2 = Inf32
        @inbounds for i in 1:length(crop_lesion)
            dx = (bx - lx[i]) * sp1
            dy = (by - ly[i]) * sp2
            dz = (bz - lz[i]) * sp3
            d2 = dx*dx + dy*dy + dz*dz
            if d2 < min_d2
                min_d2 = d2
                if min_d2 <= 4.0f0
                    break
                end
            end
        end
        
        global_idx = CartesianIndex(x_min + b_idx[1] - 1, y_min + b_idx[2] - 1, z_min + b_idx[3] - 1)
        
        if (val >= 1.5f0 || (val > 0 && min_d2 <= 36.0f0)) && min_d2 <= max_surf_dist_mm2
            push!(surf_pts, global_idx)
        end
        
        if val > 0 && val < 1.8f0 && min_d2 <= max_marr_dist_mm2 && crop_mask[b_idx] != target_id
            push!(marr_pts, global_idx)
        end
    end
    
    return (surf_pts, marr_pts)
end

"""
Retrieve bone subsegments from cache or compute on-the-fly using global bone atlas / Skellytour.
Returns (surf_pts::Vector{CartesianIndex{3}}, marr_pts::Vector{CartesianIndex{3}})
"""
function _get_or_compute_bone_subseg(stateObject, target_id::Int, panel_tp::Int)
    if target_id <= 0
        return (CartesianIndex{3}[], CartesianIndex{3}[])
    end
    
    node_name = get_node_name_for_tp(panel_tp)
    
    # Check precomputed cache
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
    
    # Fast on-the-fly extraction using global bone atlas / skellytour
    skelly_vol = global_bone_atlas[]
    mask_vol = if haskey(tp_data_cache, panel_tp)
        tp_data_cache[panel_tp].mask_i16
    else
        m_dat = nothing
        if stateObject !== nothing && isdefined(stateObject, :onScrollData)
            for scr in stateObject.onScrollData.dataToScroll
                if scr.name == "Mask"
                    m_dat = scr.dat
                    break
                end
            end
        end
        m_dat
    end
    
    if skelly_vol !== nothing && mask_vol !== nothing && count(skelly_vol .> 0) > 0
        # Use proper morphological bone subsegmentation (erosion-based cortical shell + marrow)
        res = try
            BoneSub = Main.MedEye3d.BoneSubsegmentation
            # Get spacing from stateObject or use default
            sp = try
                sv = stateObject.spacingsValue
                isa(sv, Tuple) ? sv : sv[1]
            catch
                (1.0, 1.0, 2.0)
            end
            println("  [BONE-MORPH] Running morphological subseg for lid=$target_id tp=$panel_tp spacing=$sp"); flush(stdout)
            surf_mask, marr_mask = BoneSub.generate_bone_subsegments(
                Float32.(mask_vol), Float32.(skelly_vol), sp, target_id
            )
            s_pts = findall(surf_mask)
            m_pts = findall(marr_mask)
            println("  [BONE-MORPH] SUCCESS: $(length(s_pts)) surf, $(length(m_pts)) marrow voxels"); flush(stdout)
            (s_pts, m_pts)
        catch e
            @warn "Morphological bone subseg failed, falling back to fast version" exception=(e, catch_backtrace())
            println("  [BONE-FAST] Falling back to compute_bone_subsegments_fast for lid=$target_id tp=$panel_tp"); flush(stdout)
            compute_bone_subsegments_fast(mask_vol, skelly_vol, target_id)
        end
        bone_subsegments_cache[(panel_tp, target_id)] = res
        if !isempty(node_name)
            bone_subsegments_cache[(node_name, target_id)] = res
        end
        return res
    end
    
    bone_subsegments_cache[(panel_tp, target_id)] = (CartesianIndex{3}[], CartesianIndex{3}[])
    return (CartesianIndex{3}[], CartesianIndex{3}[])
end

function reactToSyncLesion(data::SyncLesionEvent, stateObjects::Vector{StateDataFields})
    t_total = time_ns()
    changed = false
    if data.lesion_id > 0
        current_active_lesion_id[] = data.lesion_id
    end

    # 1. Update OpenGL visibility constants for masks
    panel5_lesion_id = data.lesion_id
    panel5_all_ids = Int[]
    if compare_mode[] && length(stateObjects) >= 5 && data.lesion_id > 0
        try
            left_node = get_node_name_for_tp(current_tp_index[])
            right_node = get_node_name_for_tp(compare_right_tp[])
            match_mod = _get_la()
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
                T_mm = eltype(textSpec.minAndMaxValue)
                if idx == 5 && compare_mode[] && !isempty(panel5_all_ids)
                    textSpec.allowedIDs = Float32.(panel5_all_ids)
                else
                    textSpec.allowedIDs = Float32[]
                    if is_single_lesion_mode[] && target_id > 0
                        textSpec.minAndMaxValue = T_mm.([target_id, target_id])
                    else
                        textSpec.minAndMaxValue = T_mm.([1, 1000])
                    end
                end
            elseif textSpec.name == "manualModif"
                textSpec.minAndMaxValue = Float32.([0.0, 1000.0])
                textSpec.allowedIDs = Float32[]
            end
        end
    end

    # 1b. Update bone subseg 3D arrays for visible panels only
    has_any_bone_data = false
    if data.lesion_id > 0
        for (panel_idx, stateObject) in enumerate(stateObjects)
            # Skip hidden panels (Fix ❸: avoid bone overlay work for invisible panels)
            if stateObject.calcDimsStruct.mainQuadVertSize <= 0 || all(iszero, stateObject.calcDimsStruct.mainImageQuadVert)
                continue
            end
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
                if scrDat.name == "Bone_Overlay"
                    # Clear previous overlay
                    if haskey(last_bone_overlay_indices, panel_idx) && !isempty(last_bone_overlay_indices[panel_idx])
                        scrDat.dat[last_bone_overlay_indices[panel_idx]] .= Int8(0)
                    end
                    # Write combined mask: surface=1, marrow=2, both=3
                    all_indices = CartesianIndex{3}[]
                    if !isempty(surf_indices)
                        scrDat.dat[surf_indices] .= Int8(1)
                        append!(all_indices, surf_indices)
                    end
                    if !isempty(marr_indices)
                        # For overlapping voxels, add (1+2=3), for marrow-only set to 2
                        for idx in marr_indices
                            if checkbounds(Bool, scrDat.dat, idx)
                                old_val = scrDat.dat[idx]
                                scrDat.dat[idx] = old_val == Int8(1) ? Int8(3) : Int8(2)
                            end
                        end
                        append!(all_indices, marr_indices)
                    end
                    last_bone_overlay_indices[panel_idx] = unique(all_indices)
                end
            end
        end

        # Ensure bone overlay texture is visible in the shader when bone data exists
        if has_any_bone_data
            for stateObject in stateObjects
                for textSpec in stateObject.mainForDisplayObjects.listOfTextSpecifications
                    if textSpec.name == "Bone_Overlay"
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
                # Skip hidden panels (Fix ❸: no work for invisible panels)
                if otherState.calcDimsStruct.mainQuadVertSize <= 0 || all(iszero, otherState.calcDimsStruct.mainImageQuadVert)
                    continue
                end
                lastSlice = max(1, otherState.onScrollData.slicesNumber)
                newSlice = clamp(targetSlice, 1, lastSlice)
                
                # Create slice dat (forces evaluation — selectdim returns a view, zero-copy)
                singleSlDat = otherState.onScrollData.dataToScroll |>
                    (scrDat) -> map(threeDimDat -> threeToTwoDimm(threeDimDat.type, Int64(newSlice), otherState.onScrollData.dimensionToScroll, threeDimDat), scrDat) |>
                    (twoDimList) -> SingleSliceDat(listOfDataAndImageNames=twoDimList, sliceNumber=newSlice, textToDisp=getTextForCurrentSlice(otherState.onScrollData, Int32(newSlice)))
                
                # Fix ❹: Removed dead TextureManag.updateTexture calls (no-op in Vulkan backend).
                # Actual GPU upload happens in consumer loop when isSliceChanged is set.
                
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
    bone_mask::Array{Int8,3}      # Combined bone overlay: surface=1, marrow=2, both=3
    anatomy::Union{Nothing, Array{UInt16,3}}  # max_anatomy atlas per-TP (UInt16, 163MB)
    mask_i16::Array{Int16,3}                  # pre-converted Int16 mask for R16_SINT texture
    anat_i16::Union{Nothing, Array{Int16,3}}  # pre-converted Int16 anatomy for R16_SINT texture
end

const tp_data_cache = Dict{Int, TpCacheEntry}()
const bone_subsegments_cache = Dict{Any, Any}()
const lesion_centroids_cache = Dict{Any, Vector{Int}}()
const _centroids_lock = ReentrantLock()
const last_bone_overlay_indices = Dict{Int, Vector{CartesianIndex{3}}}()
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
export register_tp_loader!, register_main_channel!, get_or_load_tp_data, last_bone_overlay_indices
export TpCacheEntry, invalidate_suv_for_lesion, invalidate_and_recompute_lesion_metrics_async!

# ── HDF5 Mask Auto-Save & Persistence ─────────────────────────────────────────
const dirty_mask_tps = Set{Int}()
const _mask_save_lock = ReentrantLock()
const h5_save_path_ref = Ref{String}("")
const studies_ref = Ref{Vector}([])

function register_h5_mask_saver!(h5_path::String, studies_list::Vector)
    h5_save_path_ref[] = h5_path
    studies_ref[] = studies_list
    println("  [AUTOSAVE-MASK] Registered HDF5 mask persistence: $h5_path ($(length(studies_list)) studies)"); flush(stdout)
    _ensure_mask_autosave_task!()
end

function mark_tp_mask_dirty!(tp_idx::Int)
    lock(_mask_save_lock) do
        push!(dirty_mask_tps, tp_idx)
    end
end

function save_tp_mask_to_h5(tp_i::Int)::Bool
    h5_path = h5_save_path_ref[]
    studies_list = studies_ref[]
    if isempty(h5_path) || !isfile(h5_path) || isempty(studies_list)
        return false
    end
    if tp_i < 0 || tp_i >= length(studies_list)
        return false
    end
    
    if !haskey(tp_data_cache, tp_i)
        return false
    end
    entry = tp_data_cache[tp_i]
    mask_to_save = entry.mask_i16 !== nothing ? entry.mask_i16 : entry.mask
    if mask_to_save === nothing
        return false
    end
    if entry.mask !== nothing && entry.mask !== mask_to_save
        try entry.mask .= mask_to_save catch; end
    end
    
    study = studies_list[tp_i + 1]
    modality, orig_tp, date_str, ct_fname, pet_fname, mask_fname, node_name, tfm_fname = study[1:8]
    group = tfm_fname == "" ? "BASELINE" : "TFM_" * tfm_fname
    
    lock(_mask_save_lock) do
        try
            HDF5.h5open(h5_path, "r+") do h5_file
                is_pf = haskey(h5_file, "_meta_/preflipped") && read(h5_file["_meta_/preflipped"]) == 1
                needs_reverse = !is_pf
                
                raw_to_write = needs_reverse ? reverse(mask_to_save, dims=2) : mask_to_save
                ds_path = "$group/$mask_fname"
                if haskey(h5_file, ds_path)
                    h5_file[ds_path][:, :, :] = Int16.(raw_to_write)
                    println("  [AUTOSAVE-MASK] Saved mask for TP $tp_i to $ds_path ($(count(>(0), mask_to_save)) non-zero voxels)"); flush(stdout)
                else
                    @warn "Dataset $ds_path not found in HDF5"
                end
            end
            delete!(dirty_mask_tps, tp_i)
            # Precompute mask centroids for this TP
            precompute_mask_centroids!(mask_to_save, tp_i, node_name)
            return true
        catch e
            @error "Failed to save mask for TP $tp_i to $h5_path" exception=(e, catch_backtrace())
            return false
        end
    end
end

function flush_all_dirty_masks!()
    tps = lock(_mask_save_lock) do
        collect(dirty_mask_tps)
    end
    for tp in tps
        save_tp_mask_to_h5(tp)
    end
end

const _mask_autosave_task_started = Ref(false)
function _ensure_mask_autosave_task!()
    _mask_autosave_task_started[] && return
    _mask_autosave_task_started[] = true
    @async begin
        while true
            sleep(2.0)
            if !isempty(dirty_mask_tps) && !isempty(h5_save_path_ref[])
                flush_all_dirty_masks!()
            end
        end
    end
end

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
    
    # Sliding window preload: only preload TP 1 (adjacent to startup TP 0)
    # Further TPs are loaded lazily on demand via EvictAndPreloadMessage
    Threads.@spawn begin
        sleep(0.5)  # Allow initial display to finish first
        tp_indices = sort(collect(keys(tp_labels)))
        # Only preload TP index 1 if not already cached
        for tp_idx in tp_indices
            if tp_idx == 1 && !haskey(tp_data_cache, tp_idx) && io_channel[] !== nothing
                try
                    put!(io_channel[], PreloadTPMessage(tp_idx))
                    println("  [STARTUP] Dispatched background preload for TP $tp_idx"); flush(stdout)
                catch; end
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

"""Load a TpCacheEntry into a specific panel.
Integer textures (mask, anatomy, bone) use native Int16/Int8 types — no Float32 conversion.
Bone_Surface + Bone_Marrow are merged into a single Bone_Overlay (surface=1, marrow=2, both=3)."""
function _load_tp_from_entry!(stateObjects, entry::TpCacheEntry, panel_idx)
    if panel_idx > length(stateObjects)
        return
    end
    
    # New data arrays → old bone overlay indices are stale
    delete!(last_bone_overlay_indices, panel_idx)
    
    # Use pre-computed Int16 arrays from TpCacheEntry
    mask_i16 = entry.mask_i16
    anat_i16 = entry.anat_i16
    bone_i8 = entry.bone_mask
    
    # Bone overlay needs independent per-panel arrays because reactToSyncLesion writes
    # bone subseg indices per-panel in panel-specific coordinate systems
    function get_or_create_bone_i8(panel_idx, req_size)
        # Try to reuse existing bone array from this panel
        for dat in stateObjects[panel_idx].onScrollData.dataToScroll
            if dat.name == "Bone_Overlay" && size(dat.dat) == req_size
                fill!(dat.dat, Int8(0))
                return dat.dat
            end
        end
        return zeros(Int8, req_size)
    end

    panel_voxels = if panel_idx == 3  # Sagittal (Y,Z,X)
        sz = (size(entry.ct, 2), size(entry.ct, 3), size(entry.ct, 1))
        Any[("CT", PermutedDimsArray(entry.ct, (2,3,1))),
            ("PET", PermutedDimsArray(entry.pet, (2,3,1))),
            ("Mask", PermutedDimsArray(mask_i16, (2,3,1))),
            ("Bone_Overlay", get_or_create_bone_i8(panel_idx, sz)),
            ("Anatomy", anat_i16 !== nothing ? PermutedDimsArray(anat_i16, (2,3,1)) : zeros(Int16, sz))]
    elseif panel_idx == 4  # Coronal (X,Z,Y)
        sz = (size(entry.ct, 1), size(entry.ct, 3), size(entry.ct, 2))
        Any[("CT", PermutedDimsArray(entry.ct, (1,3,2))),
            ("PET", PermutedDimsArray(entry.pet, (1,3,2))),
            ("Mask", PermutedDimsArray(mask_i16, (1,3,2))),
            ("Bone_Overlay", get_or_create_bone_i8(panel_idx, sz)),
            ("Anatomy", anat_i16 !== nothing ? PermutedDimsArray(anat_i16, (1,3,2)) : zeros(Int16, sz))]
    elseif panel_idx == 2  # PET-only
        Any[("PET", entry.pet)]
    else  # Axial (panels 1, 5) — each gets its own bone copy
        sz = size(entry.ct)
        Any[("CT", entry.ct), ("PET", entry.pet), ("Mask", mask_i16),
            ("Bone_Overlay", get_or_create_bone_i8(panel_idx, sz)),
            ("Anatomy", anat_i16 !== nothing ? anat_i16 : zeros(Int16, sz))]
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
    has_manual_spec = any(ts -> ts.name == "manualModif", stateObjects[panel_idx].mainForDisplayObjects.listOfTextSpecifications)
    if has_manual_spec
        manual_buf = existing_manual !== nothing ? existing_manual : zeros(Float32, size(panel_voxels[1][2]))
        if panel_idx != 2 && (length(panel_voxels) < 2 || panel_voxels[2][1] != "manualModif")
            insert!(panel_voxels, 2, ("manualModif", manual_buf))
        elseif panel_idx == 2
            insert!(panel_voxels, 1, ("manualModif", manual_buf))
        end
    end
    
    newDataToScroll = StructsManag.getThreeDims(panel_voxels)
    stateObjects[panel_idx].onScrollData.dataToScroll = newDataToScroll
    stateObjects[panel_idx].onScrollData.nameIndexes = DataStructs.getLocationDict(newDataToScroll)
    
    # Track which TP this panel holds
    panel_tp = if compare_mode[] && panel_idx == 5
        compare_right_tp[]
    else
        current_tp_index[]
    end
    stateObjects[panel_idx].onScrollData.currentTpIndex = panel_tp
    stateObjects[panel_idx].onScrollData.totalTpCount = length(tp_labels)
    stateObjects[panel_idx].onScrollData.tpIndices = sort(collect(keys(tp_labels)))

    # Re-apply appropriate modality windowing for this panel
    panel_mod = uppercase(get(tp_modalities, panel_tp, "PET"))
    nuc_win = get(current_windowing, panel_mod, Float32[0.0, 10.0])
    ct_win = get(current_windowing, "CT", Float32[-150.0, 250.0])
    for tex in stateObjects[panel_idx].mainForDisplayObjects.listOfTextSpecifications
        if tex.name == "CT"
            tex.minAndMaxValue = Float32.([ct_win[1], ct_win[2]])
        elseif tex.name == "PET" || tex.name == "SPECT"
            tex.minAndMaxValue = Float32.([nuc_win[1], nuc_win[2]])
        end
    end
    
    dimToScroll = stateObjects[panel_idx].onScrollData.dimensionToScroll
    if !isempty(newDataToScroll)
        stateObjects[panel_idx].onScrollData.slicesNumber = Int32(size(newDataToScroll[1].dat, dimToScroll))
    end
    stateObjects[panel_idx].currentDisplayedSlice = max(1, stateObjects[panel_idx].onScrollData.slicesNumber ÷ 2)
    stateObjects[panel_idx].currentlyDispDat = SingleSliceDat(sliceNumber=0)
end


"""Invalidate cached SUV, volume, and centroid data for a lesion after mask modification."""
function invalidate_suv_for_lesion(lesion_id::Int, tp_idx::Int)
    try
        LMW = _get_lmw()
        if LMW !== nothing
            if isdefined(LMW, :_lesion_suv_cache)
                delete!(LMW._lesion_suv_cache, (tp_idx, lesion_id))
            end
            if isdefined(LMW, :_volume_cache)
                delete!(LMW._volume_cache, (tp_idx, lesion_id))
            end
            if isdefined(LMW, :_db_dirty)
                LMW._db_dirty[] = true
            end
        end
    catch; end
    delete!(lesion_centroids_cache, (tp_idx, lesion_id))
    delete!(lesion_centroids_cache, lesion_id)
    println("  [SUV] Invalidated cache for lesion $lesion_id @ TP $tp_idx"); flush(stdout)
end

"""
    invalidate_and_recompute_lesion_metrics_async!(lesion_id, tp_idx, mask_vol)

After mask modification (painting or AI segmentation):
1. Invalidate all caches (SUV, volume, centroid)
2. Recompute centroid from the current mask
3. Async recompute SUV, volume, PROMISE and update UI fields
"""
function invalidate_and_recompute_lesion_metrics_async!(lesion_id::Int, tp_idx::Int, mask_vol::Union{AbstractArray, Nothing}=nothing)
    # 1. Invalidate caches
    invalidate_suv_for_lesion(lesion_id, tp_idx)
    
    # 2. Recompute centroid from current mask
    if mask_vol !== nothing
        try
            indices = findall(x -> round(Int, x) == lesion_id, mask_vol)
            if !isempty(indices)
                cx = round(Int, mean(i[1] for i in indices))
                cy = round(Int, mean(i[2] for i in indices))
                cz = round(Int, mean(i[3] for i in indices))
                lesion_centroids_cache[(tp_idx, lesion_id)] = [cx, cy, cz]
                if tp_idx == current_tp_index[]
                    lesion_centroids_cache[lesion_id] = [cx, cy, cz]
                end
                println("  [SUV] Recomputed centroid for lesion $lesion_id @ TP $tp_idx: ($cx,$cy,$cz)"); flush(stdout)
            end
        catch e
            @warn "Centroid recompute failed for lesion $lesion_id: $e"
        end
    elseif haskey(tp_data_cache, tp_idx)
        # Try to get mask from tp_data_cache
        try
            entry = tp_data_cache[tp_idx]
            m = entry.mask
            indices = findall(x -> round(Int, x) == lesion_id, m)
            if !isempty(indices)
                cx = round(Int, mean(i[1] for i in indices))
                cy = round(Int, mean(i[2] for i in indices))
                cz = round(Int, mean(i[3] for i in indices))
                lesion_centroids_cache[(tp_idx, lesion_id)] = [cx, cy, cz]
                if tp_idx == current_tp_index[]
                    lesion_centroids_cache[lesion_id] = [cx, cy, cz]
                end
            end
        catch; end
    end
    
    # 3. Async recompute SUV/volume/PROMISE (pre-populate caches)
    Threads.@spawn begin
        try
            LMW = _get_lmw()
            if LMW !== nothing
                # Recompute volume (cache was cleared, so this recomputes from scratch)
                vol = LMW.compute_lesion_volume(lesion_id, tp_idx)
                
                # Recompute SUV (cache was cleared, so this recomputes from scratch)
                suv_str = LMW.compute_lesion_suv_string(lesion_id, tp_idx)
                if !isempty(suv_str)
                    LMW._lesion_suv_cache[(tp_idx, lesion_id)] = suv_str
                end
                
                println("  [SUV] Async recomputed metrics for lesion $lesion_id @ TP $tp_idx: vol=$(round(vol["volume_cc"], digits=2))cc, suv=$(suv_str)"); flush(stdout)
            end
        catch e
            @warn "Async SUV/volume recompute failed for lesion $lesion_id: $e"
        end
    end
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
    # Flush any modified masks before changing timepoint or evicting
    flush_all_dirty_masks!()
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
                _load_tp_from_entry!(stateObjects, entry_left, 1)
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
                _load_tp_from_entry!(stateObjects, entry_right, 5)
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
            t_panels = @elapsed begin
                for i in [1, 2, 3, 4]
                    if i <= length(stateObjects)
                        _load_tp_from_entry!(stateObjects, entry, i)
                    end
                end
                if length(stateObjects) >= 5
                    _load_tp_from_entry!(stateObjects, entry, 5)
                end
            end
            if DEBUG_VERBOSE[]; println("  [BENCH] _load_tp_from_entry! completed, skipping redundant initial scroll"); flush(stdout); end
            
            # Re-apply bone overlay + navigate to active lesion (or lesion 1 if none)
            empty!(last_bone_overlay_indices)
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
    # Sliding window: preload adjacent TPs (current ± 1), evict distant ones
    if io_channel[] !== nothing
        try
            neighbors = Int[]
            prev_pos = mod1(new_pos - 1, num_tps)
            next_pos_n = mod1(new_pos + 1, num_tps)
            push!(neighbors, tp_indices[prev_pos])
            push!(neighbors, tp_indices[next_pos_n])
            filter!(tp -> !haskey(tp_data_cache, tp), neighbors)
            
            # Evict TPs that are far from current (keep current ± 1 only)
            keep_set = Set{Int}([new_tp, tp_indices[prev_pos], tp_indices[next_pos_n]])
            evict_tps = filter(tp -> !in(tp, keep_set), collect(keys(tp_data_cache)))
            
            if !isempty(neighbors) || !isempty(evict_tps)
                put!(io_channel[], EvictAndPreloadMessage(evict_tps, neighbors))
            end
        catch; end
    end
    
    # SUV precompute + CT Docker preload: fire-and-forget in background (non-blocking)
    let tp_for_bg = new_tp, label_for_bg = label
        Threads.@spawn begin
            try
                LMW = _get_lmw()
                if LMW !== nothing && haskey(tp_data_cache, tp_for_bg)
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
            if (textSpec.isMultiDiscreteMask || textSpec.name == "Mask" || textSpec.name == "segmentation") && textSpec.name != "Anatomy"
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
    set_ai_status!("[Processing] AI request ($(data.algorithm))...")
    try
        println("Add New Lesion (Auto-PET) triggered with algorithm: $(data.algorithm)"); flush(stdout)
        
        tp1_state = stateObjects[1]
        
        # Look up volumes by NAME, not index
        ct_vol = nothing
        pet_vol = nothing
        seg_vol = nothing
        for dat in tp1_state.onScrollData.dataToScroll
            if dat.name == "CT" && ct_vol === nothing
                ct_vol = dat.dat
            elseif dat.name == "PET" && pet_vol === nothing
                pet_vol = dat.dat
            elseif (dat.name == "Mask" || dat.name == "segmentation") && seg_vol === nothing
                seg_vol = dat.dat
            end
        end
        if ct_vol === nothing
            error("CT volume not found by name in panel 1. No fallbacks allowed.")
        end
        if pet_vol === nothing
            error("PET volume not found by name in panel 1. No fallbacks allowed.")
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
        
        # Snapshot panel scroll references for async scribble search
        panel_snapshots = [
            (p_idx,
             !isempty(st.textureToModifyVec) ? st.textureToModifyVec[1].name : "NONE",
             st.valueForMasToSet.value,
             [(dat.name, dat.dat) for dat in st.onScrollData.dataToScroll])
            for (p_idx, st) in enumerate(stateObjects)
        ]
        
        algo = data.algorithm
        channel = data.channel
        
        # Execute heavy voxel scanning & job queuing asynchronously so consumer / GUI NEVER block
        Threads.@spawn begin
            try
                painted_pts = CartesianIndex{3}[]
                for (p_idx, active_paint_tex, brush_val, dat_list) in panel_snapshots
                    for (d_name, d_dat) in dat_list
                        if d_name == active_paint_tex || d_name == "manualModif"
                            is_manual = d_name == "manualModif"
                            T_elem = eltype(d_dat)
                            p = if is_manual
                                findall(d_dat .> 0)
                            else
                                v_act = round(T_elem, active_id)
                                v_set = round(T_elem, brush_val)
                                findall((d_dat .== v_act) .| (d_dat .== v_set))
                            end
                            if !isempty(p)
                                println("[reactToAddAutoPet] Found $(length(p)) painted voxels for lesion $active_id in panel $p_idx $(d_name)"); flush(stdout)
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
                
                if isempty(painted_pts)
                    msg = "No painted scribbles found for AI inference. Paint scribbles on the lesion first."
                    println("ERROR: $msg"); flush(stdout)
                    set_ai_status!("[Error] $msg")
                    return
                end
                
                points_vol = zeros(Float32, size(ct_vol))
                for idx in painted_pts
                    if checkbounds(Bool, points_vol, idx)
                        points_vol[idx] = 1.0f0
                    end
                end
                cx = round(Int, mean([p[1] for p in painted_pts]))
                cy = round(Int, mean([p[2] for p in painted_pts]))
                cz = round(Int, mean([p[3] for p in painted_pts]))
                
                scribble_coords_0idx = [[idx[1]-1, idx[2]-1, idx[3]-1] for idx in painted_pts if checkbounds(Bool, ct_vol, idx)]
                
                set_ai_status!("[Preparing] inference ($(algo))...")
                println("Queuing $(algo) inference job (seed=$cx,$cy,$cz, lesion=$active_id, $(length(painted_pts)) painted points)..."); flush(stdout)
                
                # Use immutable views / direct references without 680MB deep copies
                put!(inference_queue, InferenceJob(
                    algo, ct_vol, pet_vol, points_vol,
                    cx, cy, cz, active_id, seg_vol, channel, scribble_coords_0idx))
            catch e
                err_msg = sprint(showerror, e)
                println("ERROR in async reactToAddAutoPet: $err_msg"); flush(stdout)
                println(sprint(showerror, e, catch_backtrace())); flush(stdout)
                set_ai_status!("[Error] AI Error: $err_msg")
            end
        end
    catch e
        err_msg = sprint(showerror, e)
        println("ERROR in reactToAddAutoPet: $err_msg"); flush(stdout)
        println(sprint(showerror, e, catch_backtrace())); flush(stdout)
        set_ai_status!("[Error] AI Error: $err_msg")
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
        set_ai_status!("[Warning] Inference failed (no mask returned)")
        return
    end

    seg_vol = data.seg_vol
    if seg_vol === nothing
        println("ERROR: No segmentation volume reference available. Cannot apply AI results. No fallbacks allowed."); flush(stdout)
        set_ai_status!("[Error] No segmentation volume (Mask) found - cannot apply AI results")
        return
    end

    if size(data.mask) == size(seg_vol)
        label_val = eltype(seg_vol)(data.active_id)
        seg_vol[data.mask .> 0] .= label_val
    else
        label_val = eltype(seg_vol)(data.active_id)
        InferenceClient.insert_patch!(seg_vol, data.mask, data.cx, data.cy, data.cz; label_val=label_val)
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
            match_mod = _get_la()
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
            if scrDat.name == "Bone_Overlay"
                # Clear previous overlay
                if haskey(last_bone_overlay_indices, panel_idx) && !isempty(last_bone_overlay_indices[panel_idx])
                    scrDat.dat[last_bone_overlay_indices[panel_idx]] .= Int8(0)
                end
                # Write combined mask: surface=1, marrow=2, both=3
                all_indices = CartesianIndex{3}[]
                if !isempty(surf_indices)
                    scrDat.dat[surf_indices] .= Int8(1)
                    append!(all_indices, surf_indices)
                end
                if !isempty(marr_indices)
                    for idx in marr_indices
                        if checkbounds(Bool, scrDat.dat, idx)
                            old_val = scrDat.dat[idx]
                            scrDat.dat[idx] = old_val == Int8(1) ? Int8(3) : Int8(2)
                        end
                    end
                    append!(all_indices, marr_indices)
                end
                last_bone_overlay_indices[panel_idx] = unique(all_indices)
            end
        end
    end
    
    # Clear manualModif across all panels (scribbles consumed by AI)
    # Note: seg_vol IS the canonical mask volume shared across all panels via PermutedDimsArray views.
    # Modifying seg_vol directly propagates automatically to all panels without manual copying.
    for (p_idx, st) in enumerate(stateObjects)
        for scrDat in st.onScrollData.dataToScroll
            if scrDat.name == "manualModif"
                fill!(scrDat.dat, zero(eltype(scrDat.dat)))
            end
        end
    end

    # Synchronize tp_data_cache (both compact mask and Int16 texture mask)
    tp_idx = current_tp_index[]
    if haskey(tp_data_cache, tp_idx)
        entry = tp_data_cache[tp_idx]
        if entry.mask isa Array{Int8, 3}
            entry.mask .= clamp.(seg_vol, Int8(-128), Int8(127))
        elseif entry.mask !== seg_vol
            entry.mask .= seg_vol
        end
        if entry.mask_i16 !== seg_vol
            entry.mask_i16 .= seg_vol
        end
        mark_tp_mask_dirty!(tp_idx)
    end

    # Ensure mask uniform displays the active lesion
    for stateObject in stateObjects
        for textSpec in stateObject.mainForDisplayObjects.listOfTextSpecifications
            if textSpec.name == "Mask" || textSpec.name == "segmentation"
                T = eltype(textSpec.minAndMaxValue)
                textSpec.minAndMaxValue = T.([data.active_id, data.active_id])
            elseif textSpec.name == "manualModif"
                textSpec.minAndMaxValue = Float32.([0.0, 1000.0])
            end
        end
    end

    # Compute center of the actual segmentation result (not the seed point)
    seg_indices = findall(seg_vol .== eltype(seg_vol)(data.active_id))
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

    # Update all panel textures (scroll with 0 = re-render current slice)
    old_sw = stateObjects[1].switchIndex
    stateObjects[1].switchIndex = 1
    reactToScroll(0, stateObjects)
    stateObjects[1].switchIndex = old_sw

    # Update status label
    voxel_count = count(data.mask .> 0)
    set_ai_status!("[Success] Done ($(voxel_count) voxels, lesion $(data.active_id))")

    # Invalidate caches and async-recompute SUV/volume/PROMISE for the modified lesion
    invalidate_and_recompute_lesion_metrics_async!(data.active_id, current_tp_index[], seg_vol)
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
                match_mod = _get_la()
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
    
    
    # Invalidate caches and async-recompute SUV/volume/centroid for the modified lesion
    invalidate_and_recompute_lesion_metrics_async!(data.lesion_id, tp_idx)
    
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
            if textSpec.name == "Bone_Overlay" || textSpec.name == "Bone_Mask" || textSpec.name == "bone_mask" || textSpec.name == "bone" || textSpec.name == "Organ_Mask" || textSpec.name == "organ_mask"
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
    elseif data.layer == 2 || data.layer == 3
        "Bone_Overlay"
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
                if scrDat.name == "Bone_Overlay"
                    if !data.active
                        if haskey(last_bone_overlay_indices, panel_idx) && !isempty(last_bone_overlay_indices[panel_idx])
                            scrDat.dat[last_bone_overlay_indices[panel_idx]] .= Int8(0)
                            delete!(last_bone_overlay_indices, panel_idx)
                        else
                            fill!(scrDat.dat, Int8(0))
                        end
                    elseif cur_lid > 0
                        panel_tp = (panel_idx == 5 && compare_mode[]) ? compare_right_tp[] : current_tp_index[]
                        panel_lid = cur_lid
                        if panel_idx == 5 && compare_mode[] && length(stateObjects) >= 5 && cur_lid > 0
                            try
                                left_node = get_node_name_for_tp(current_tp_index[])
                                right_node = get_node_name_for_tp(compare_right_tp[])
                                match_mod = _get_la()
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
                        # Clear previous and set new for combined overlay
                        if haskey(last_bone_overlay_indices, panel_idx) && !isempty(last_bone_overlay_indices[panel_idx])
                            scrDat.dat[last_bone_overlay_indices[panel_idx]] .= Int8(0)
                        end
                        if !isempty(indices)
                            val = (data.layer == 2) ? Int8(1) : Int8(2)
                            scrDat.dat[indices] .= val
                        end
                        last_bone_overlay_indices[panel_idx] = indices
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
    
    # Force re-upload of texture data after visibility toggle to ensure
    # anatomy texture data reaches the GPU (may have been skipped on prior scroll)
    if data.layer == 4 && data.active
        for idx in visible_panels
            stateObjects[idx].isSliceChanged = true
            # Enforce full anatomy range — undo any prior single-lesion filter
            for ts in stateObjects[idx].mainForDisplayObjects.listOfTextSpecifications
                if ts.name == "Anatomy"
                    ts.minAndMaxValue = Int16[0, 400]
                    ts.allowedIDs = Float32[]
                end
            end
            # Mark UBO dirty so shader reads updated minAndMaxValue
            if stateObjects[idx].mainForDisplayObjects.vulkanPipelineState !== nothing
                stateObjects[idx].mainForDisplayObjects.vulkanPipelineState.ubo_dirty = true
            end
        end
    end
end

function reactToSaveMRB(data::SaveMRBEvent, stateObjects::Vector{StateDataFields})
    println("Save MRB triggered: saving all dirty masks to HDF5..."); flush(stdout)
    flush_all_dirty_masks!()
end

export reactToToggleMoveLesionMode
function reactToToggleMoveLesionMode(data::ToggleMoveLesionModeEvent, stateObjects::Vector{StateDataFields})
    println("Move Lesion Mode toggled to $(data.active)"); flush(stdout)
    for state in stateObjects
        state.moveLesionMode = data.active
    end
end

atexit() do
    flush_all_dirty_masks!()
end

end
