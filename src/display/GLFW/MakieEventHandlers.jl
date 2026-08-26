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
    pts = findall(dat .== lesion_id)
    if isempty(pts)
        pts = findall(abs.(dat .- lesion_id) .< 0.1f0)
    end
    if isempty(pts)
        return nothing
    end
    sum_x = sum(p -> p[1], pts)
    sum_y = sum(p -> p[2], pts)
    sum_z = sum(p -> p[3], pts)
    n = length(pts)
    return [round(Int, sum_x / n), round(Int, sum_y / n), round(Int, sum_z / n)]
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
                    rm = Float32.(entry.mask)
                    rs = Float32.(entry.bone_surf)
                    rmr = Float32.(entry.bone_marr)
                    _load_tp_from_entry!(stateObjects, entry, 5; mask_f32=rm, bone_s_f32=rs, bone_m_f32=rmr)
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
            
            left_label = get(tp_labels, current_tp_index[], "TP $(current_tp_index[])")
            right_label = get(tp_labels, compare_right_tp[], "TP $(compare_right_tp[])")
            println("Compare mode ON: Left=$left_label, Right=$right_label"); flush(stdout)
            
            # Re-apply bone overlay for active lesion
            if current_active_lesion_id[] > 0
                reactToSyncLesion(SyncLesionEvent(current_active_lesion_id[]), stateObjects)
            end
        else
            compare_right_tp[] = -1
            # Reload current active TP into all 4 panels using _load_tp_from_entry!
            entry = get_or_load_tp_data(current_tp_index[])
            if entry !== nothing
                m = Float32.(entry.mask)
                s = Float32.(entry.bone_surf)
                mr = Float32.(entry.bone_marr)
                num_panels = min(4, length(stateObjects))
                for i in 1:num_panels
                    _load_tp_from_entry!(stateObjects, entry, i; mask_f32=m, bone_s_f32=s, bone_m_f32=mr)
                end
            end
            # Evict inactive TPs from RAM
            for k in collect(keys(tp_data_cache))
                if k != current_tp_index[]
                    delete!(tp_data_cache, k)
                end
            end
            GC.gc(false)

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
            
            # Re-apply bone overlay for active lesion
            if current_active_lesion_id[] > 0
                reactToSyncLesion(SyncLesionEvent(current_active_lesion_id[]), stateObjects)
            end
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
            # Update nuclear overlay contribution (PET/SPECT, not the pure PET main image panel)
            if tex.isNuclearMask && !tex.isMainImage
                Uniforms.setTextureContribution(tex, data.weight)
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
    @warn "No node name for TP $tp_idx — tp_node_names not populated from HDF5"
    return "Unknown_TP_$tp_idx"
end

const current_active_lesion_id = Ref(0)

"""
Compute or retrieve cached bone subsegments for a given stateObject + lesion + TP.
Uses Bool thresholding instead of Float32 to save ~1GB allocation.
Returns (surf_pts::Vector{CartesianIndex{3}}, marr_pts::Vector{CartesianIndex{3}})
"""
function _get_or_compute_bone_subseg(stateObject, target_id::Int, panel_tp::Int)
    if target_id <= 0
        return (CartesianIndex{3}[], CartesianIndex{3}[])
    end
    
    # Check cache
    cached = if haskey(bone_subsegments_cache, (panel_tp, target_id))
        bone_subsegments_cache[(panel_tp, target_id)]
    elseif haskey(bone_subsegments_cache, (get_node_name_for_tp(panel_tp), target_id))
        bone_subsegments_cache[(get_node_name_for_tp(panel_tp), target_id)]
    else
        nothing
    end
    
    if cached !== nothing
        if cached === :computing
            println("  [BONE-DEBUG] target_id=$target_id (tp=$panel_tp): still :computing"); flush(stdout)
            return (CartesianIndex{3}[], CartesianIndex{3}[])
        end
        raw_surf, raw_marr = cached
        surf_res = raw_surf isa AbstractArray{<:CartesianIndex} ? raw_surf : findall(raw_surf .> 0)
        marr_res = raw_marr isa AbstractArray{<:CartesianIndex} ? raw_marr : findall(raw_marr .> 0)
        println("  [BONE-DEBUG] target_id=$target_id (tp=$panel_tp): CACHE HIT ($(length(surf_res)) surf, $(length(marr_res)) marr)"); flush(stdout)
        return (surf_res, marr_res)
    end
    
    # Compute fresh
    panel_seg = nothing
    panel_ct = nothing
    for dat in stateObject.onScrollData.dataToScroll
        if dat.name == "Mask" || dat.name == "segmentation"
            panel_seg = dat.dat
        elseif dat.name == "CT"
            panel_ct = dat.dat
        end
    end
    
    if panel_seg === nothing
        return (CartesianIndex{3}[], CartesianIndex{3}[])
    end
    
    # Bool threshold instead of Float32 (saves ~1GB allocation)
    local_bone_atlas = panel_ct !== nothing ? (panel_ct .> 150.0f0) : global_bone_atlas[]
    if local_bone_atlas === nothing
        return (CartesianIndex{3}[], CartesianIndex{3}[])
    end
    
    panel_lesion_indices = findall(panel_seg .== Float32(target_id))
    has_bone = !isempty(panel_lesion_indices) && 
        any(idx -> checkbounds(Bool, local_bone_atlas, idx) && local_bone_atlas[idx] > 0, panel_lesion_indices)
    
    if has_bone
        # Mark as computing to avoid redundant spawns
        bone_subsegments_cache[(panel_tp, target_id)] = :computing
        
        # We must copy or reference the arrays safely for the background task
        bg_panel_seg = copy(panel_seg)
        bg_bone_atlas = copy(local_bone_atlas)
        
        Threads.@spawn begin
            try
                println("  [ASYNC-BONE] Starting background thread for lesion $target_id (tp=$panel_tp)")
                t_bg = time_ns()
                
                s_mask, m_mask = Main.MedEye3d.BoneSubsegmentation.generate_bone_subsegments(
                    bg_panel_seg, Float32.(bg_bone_atlas), (1.5f0, 1.5f0, 2.0f0), target_id)
                    
                pts_surf = findall(s_mask)
                pts_marr = findall(m_mask)
                
                t_bg_ms = (time_ns() - t_bg) / 1e6
                println("  [ASYNC-BONE] Finished in $(round(t_bg_ms, digits=1))ms. Triggering re-render.")
                
                # Send the result back to the main thread via the channel to safely update the Dict
                if main_event_channel[] !== nothing
                    put!(main_event_channel[], BoneSubsegResultEvent(panel_tp, target_id, pts_surf, pts_marr))
                else
                    # Fallback (unsafe)
                    bone_subsegments_cache[(panel_tp, target_id)] = (pts_surf, pts_marr)
                end
            catch e
                @warn "Background bone computation failed" e
                if main_event_channel[] !== nothing
                    put!(main_event_channel[], BoneSubsegResultEvent(panel_tp, target_id, CartesianIndex{3}[], CartesianIndex{3}[]))
                else
                    bone_subsegments_cache[(panel_tp, target_id)] = (CartesianIndex{3}[], CartesianIndex{3}[])
                end
            end
        end
        return (CartesianIndex{3}[], CartesianIndex{3}[])
    else
        bone_subsegments_cache[(panel_tp, target_id)] = (CartesianIndex{3}[], CartesianIndex{3}[])
        return (CartesianIndex{3}[], CartesianIndex{3}[])
    end
end

function reactToSyncLesion(data::SyncLesionEvent, stateObjects::Vector{StateDataFields})
    t_total = time_ns()
    println("  [BENCH-SL] reactToSyncLesion($(data.lesion_id)) START"); flush(stdout)
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
            elseif textSpec.name == "Bone_Surface" || textSpec.name == "Bone_Marrow"
                textSpec.isVisible = true
                ModernGL.glUseProgram(stateObject.mainForDisplayObjects.shader_program)
                Uniforms.setTextureVisibility(true, textSpec.uniforms)
                Uniforms.setMaskColor(textSpec.color, textSpec.uniforms)
            end
        end
    end
    t_uniforms_ms = (time_ns() - t_total) / 1e6
    println("  [BENCH-SL] uniforms+cross-tp: $(round(t_uniforms_ms, digits=1))ms"); flush(stdout)



    # ── Pre-compute bone subseg ONCE per unique TP (deduplicated) ──
    t_bone = time_ns()
    left_tp = current_tp_index[]
    left_axial_surf, left_axial_marr = try
        _get_or_compute_bone_subseg(stateObjects[1], data.lesion_id, left_tp)
    catch e
        println("Failed to compute bone subseg for left TP: $e")
        (CartesianIndex{3}[], CartesianIndex{3}[])
    end

    right_axial_surf, right_axial_marr = if compare_mode[] && length(stateObjects) >= 5
        try
            _get_or_compute_bone_subseg(stateObjects[5], panel5_lesion_id, compare_right_tp[])
        catch e
            println("Failed to compute bone subseg for right TP: $e")
            (CartesianIndex{3}[], CartesianIndex{3}[])
        end
    else
        (CartesianIndex{3}[], CartesianIndex{3}[])
    end
    t_bone_ms = (time_ns() - t_bone) / 1e6
    println("  [BENCH-SL] bone_subseg: $(round(t_bone_ms, digits=1))ms (left=$(length(left_axial_surf))+$(length(left_axial_marr)), right=$(length(right_axial_surf))+$(length(right_axial_marr)))"); flush(stdout)

    # ── Per-panel: just remap orientations from pre-computed axial data ──
    t_panel = time_ns()
    for (panel_idx, stateObject) in enumerate(stateObjects)
        is_right = (panel_idx == 5 && compare_mode[])
        base_surf = is_right ? right_axial_surf : left_axial_surf
        base_marr = is_right ? right_axial_marr : left_axial_marr

        surf_indices, marr_indices = if panel_idx == 3  # Sagittal (Y, Z, X)
            ([CartesianIndex(I[2], I[3], I[1]) for I in base_surf],
             [CartesianIndex(I[2], I[3], I[1]) for I in base_marr])
        elseif panel_idx == 4  # Coronal (X, Z, Y)
            ([CartesianIndex(I[1], I[3], I[2]) for I in base_surf],
             [CartesianIndex(I[1], I[3], I[2]) for I in base_marr])
        else  # Axial (panels 1, 2, 5)
            (base_surf, base_marr)
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
                println("  [BONE-DEBUG] Panel $panel_idx: Bone_Surface wrote $(length(surf_indices)) indices"); flush(stdout)
            elseif scrDat.name == "Bone_Marrow"
                if haskey(last_bone_marr_indices, panel_idx) && !isempty(last_bone_marr_indices[panel_idx])
                    scrDat.dat[last_bone_marr_indices[panel_idx]] .= 0.0f0
                end
                if !isempty(marr_indices)
                    scrDat.dat[marr_indices] .= 1.0f0
                end
                last_bone_marr_indices[panel_idx] = marr_indices
                println("  [BONE-DEBUG] Panel $panel_idx: Bone_Marrow wrote $(length(marr_indices)) indices"); flush(stdout)
            end
        end
    end
    t_panel_ms = (time_ns() - t_panel) / 1e6
    println("  [BENCH-SL] panel_remap+write: $(round(t_panel_ms, digits=1))ms"); flush(stdout)

    t_scroll = time_ns()
    active_panel_indices = if compare_mode[] && length(stateObjects) >= 5
        [1, 5]
    elseif length(stateObjects) >= 4
        [1, 2, 3, 4]
    else
        collect(1:length(stateObjects))
    end

    panel_tp_cur = current_tp_index[]
    canonical_center = if data.lesion_id > 0
        if haskey(lesion_centroids_cache, (panel_tp_cur, data.lesion_id))
            lesion_centroids_cache[(panel_tp_cur, data.lesion_id)]
        elseif haskey(lesion_centroids_cache, (get_node_name_for_tp(panel_tp_cur), data.lesion_id))
            lesion_centroids_cache[(get_node_name_for_tp(panel_tp_cur), data.lesion_id)]
        elseif haskey(lesion_centroids_cache, data.lesion_id)
            lesion_centroids_cache[data.lesion_id]
        else
            # on-the-fly centroid computation when not pre-cached
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
    println("canonical_center=$canonical_center, active_panel_indices=$active_panel_indices"); flush(stdout)

    if canonical_center !== nothing
        for idx in active_panel_indices
            stateObject = stateObjects[idx]
            last_sl = max(1, stateObject.onScrollData.slicesNumber)
            
            # Check if this panel has its own lesion center (especially in compare mode)
            target_id = (idx == 5 && compare_mode[]) ? panel5_lesion_id : data.lesion_id
            panel_center = if idx == 5 && compare_mode[] && target_id > 0
                r_tp = compare_right_tp[]
                if haskey(lesion_centroids_cache, (r_tp, target_id))
                    lesion_centroids_cache[(r_tp, target_id)]
                elseif haskey(lesion_centroids_cache, (get_node_name_for_tp(r_tp), target_id))
                    lesion_centroids_cache[(get_node_name_for_tp(r_tp), target_id)]
                elseif haskey(lesion_centroids_cache, target_id)
                    lesion_centroids_cache[target_id]
                else
                    nothing
                end
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
    t_scroll_ms = (time_ns() - t_scroll) / 1e6
    println("  [BENCH-SL] centroid+scroll: $(round(t_scroll_ms, digits=1))ms"); flush(stdout)
    t_total_ms = (time_ns() - t_total) / 1e6
    println("  [BENCH-SL] SYNC LESION TOTAL: $(round(t_total_ms, digits=1))ms"); flush(stdout)
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
    
    # Eagerly preload neighbor TPs (TP 1, TP 2) so clicking TP>> is instant
    Threads.@spawn begin
        sleep(0.5)  # Allow initial display to finish first
        tp_indices = sort(collect(keys(tp_labels)))
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

"""Load a TpCacheEntry into a specific panel, using PermutedDimsArray for sag/cor views.
Pre-converted Float32 arrays can be passed to avoid redundant conversions across panels."""
function _load_tp_from_entry!(stateObjects, entry::TpCacheEntry, panel_idx;
                              mask_f32=nothing, bone_s_f32=nothing, bone_m_f32=nothing)
    if panel_idx > length(stateObjects)
        return
    end
    
    # Convert compact types to Float32 only if not pre-supplied
    if mask_f32 === nothing
        mask_f32 = Float32.(entry.mask)
    end
    if bone_s_f32 === nothing
        bone_s_f32 = Float32.(entry.bone_surf)
    end
    if bone_m_f32 === nothing
        bone_m_f32 = Float32.(entry.bone_marr)
    end
    
    # Use PermutedDimsArray for zero-copy views
    panel_voxels = if panel_idx == 3  # Sagittal (Y,Z,X)
        Any[("CT", PermutedDimsArray(entry.ct, (2,3,1))),
            ("PET", PermutedDimsArray(entry.pet, (2,3,1))),
            ("Mask", PermutedDimsArray(mask_f32, (2,3,1))),
            ("Bone_Surface", PermutedDimsArray(bone_s_f32, (2,3,1))),
            ("Bone_Marrow", PermutedDimsArray(bone_m_f32, (2,3,1)))]
    elseif panel_idx == 4  # Coronal (X,Z,Y)
        Any[("CT", PermutedDimsArray(entry.ct, (1,3,2))),
            ("PET", PermutedDimsArray(entry.pet, (1,3,2))),
            ("Mask", PermutedDimsArray(mask_f32, (1,3,2))),
            ("Bone_Surface", PermutedDimsArray(bone_s_f32, (1,3,2))),
            ("Bone_Marrow", PermutedDimsArray(bone_m_f32, (1,3,2)))]
    elseif panel_idx == 2  # PET-only
        Any[("PET", entry.pet)]
    else  # Axial (panels 1, 5)
        Any[("CT", entry.ct), ("PET", entry.pet), ("Mask", mask_f32),
            ("Bone_Surface", bone_s_f32), ("Bone_Marrow", bone_m_f32)]
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
    stateObjects[panel_idx].currentlyDispDat = SingleSliceDat()
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
const tp_descriptions = Dict{Int, String}() # tp_index -> radiological description

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

export tp_data_cache, bone_subsegments_cache, lesion_centroids_cache, global_bone_atlas, global_organ_mapping, current_tp_index, tp_labels, tp_descriptions
export compare_mode, compare_right_tp
export pet_volumes_cache, global_ts_atlas, global_ts_names, patient_id, h5_path_ref, tp_modalities, volume_z_size


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
        println("  [BENCH] get_or_load_tp_data(left): $(round(t_load, digits=3))s"); flush(stdout)
        
        t_panel_left = @elapsed begin
            if entry_left !== nothing
                # Pre-convert once for left panel
                lm = Float32.(entry_left.mask)
                ls = Float32.(entry_left.bone_surf)
                lmr = Float32.(entry_left.bone_marr)
                _load_tp_from_entry!(stateObjects, entry_left, 1; mask_f32=lm, bone_s_f32=ls, bone_m_f32=lmr)
            end
        end
        println("  [BENCH] _load_tp_from_entry!(left): $(round(t_panel_left*1000, digits=1))ms"); flush(stdout)
        
        # Right panel: next TP chronologically
        next_pos = mod1(new_pos + 1, num_tps)
        right_tp = tp_indices[next_pos]
        compare_right_tp[] = right_tp
        
        t_load_r = @elapsed begin
            entry_right = get_or_load_tp_data(right_tp)
        end
        println("  [BENCH] get_or_load_tp_data(right): $(round(t_load_r, digits=3))s"); flush(stdout)
        
        t_panel_right = @elapsed begin
            if entry_right !== nothing
                # Pre-convert once for right panel
                rm = Float32.(entry_right.mask)
                rs = Float32.(entry_right.bone_surf)
                rmr = Float32.(entry_right.bone_marr)
                _load_tp_from_entry!(stateObjects, entry_right, 5; mask_f32=rm, bone_s_f32=rs, bone_m_f32=rmr)
            end
        end
        println("  [BENCH] _load_tp_from_entry!(right): $(round(t_panel_right*1000, digits=1))ms"); flush(stdout)
        
        # Re-render both panels
        t_render = @elapsed begin
            old_idx = stateObjects[1].switchIndex
            stateObjects[1].switchIndex = 1
            ReactToScroll.reactToScroll(0, stateObjects, false)
            stateObjects[1].switchIndex = 5
            ReactToScroll.reactToScroll(0, stateObjects, false)
            stateObjects[1].switchIndex = old_idx
        end
        println("  [BENCH] reactToScroll ×2: $(round(t_render*1000, digits=1))ms"); flush(stdout)
        
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
        println("  [BENCH] get_or_load_tp_data: $(round(t_load, digits=3))s"); flush(stdout)
        
        if entry !== nothing
            num_panels = min(length(stateObjects), 5)
            t_panels = @elapsed begin
                # Pre-convert arrays ONCE for all panels (saves ~20GB allocation)
                mask_f32 = Float32.(entry.mask)
                bone_s_f32 = Float32.(entry.bone_surf)
                bone_m_f32 = Float32.(entry.bone_marr)
                for i in [1, 2, 3, 4]
                    if i <= length(stateObjects)
                        _load_tp_from_entry!(stateObjects, entry, i;
                            mask_f32=mask_f32, bone_s_f32=bone_s_f32, bone_m_f32=bone_m_f32)
                    end
                end
                # Panel 5 (compare right) also uses same entry in non-compare mode
                if length(stateObjects) >= 5
                    _load_tp_from_entry!(stateObjects, entry, 5;
                        mask_f32=mask_f32, bone_s_f32=bone_s_f32, bone_m_f32=bone_m_f32)
                end
            end
            println("  [BENCH] _load_tp_from_entry! ×panels (PermutedDimsArray): $(round(t_panels*1000, digits=1))ms"); flush(stdout)
            
            # Re-render all panels
            t_render = @elapsed begin
                old_idx = stateObjects[1].switchIndex
                for idx in 1:min(length(stateObjects), 5)
                    stateObjects[1].switchIndex = idx
                    ReactToScroll.reactToScroll(0, stateObjects, false)
                end
                stateObjects[1].switchIndex = old_idx
            end
            println("  [BENCH] reactToScroll ×panels: $(round(t_render*1000, digits=1))ms"); flush(stdout)
            
            # Re-apply bone overlay for active lesion after TP data replacement
            if current_active_lesion_id[] > 0
                reactToSyncLesion(SyncLesionEvent(current_active_lesion_id[]), stateObjects)
            end
        end
    end
    # Dispatch eviction + preload to IO channel (non-blocking)
    needed_tps = Set([new_tp])
    preload_tps = Int[]
    if num_tps > 1
        prev_tp = tp_indices[mod1(new_pos - 1, num_tps)]
        next_tp_idx = tp_indices[mod1(new_pos + 1, num_tps)]
        push!(needed_tps, prev_tp, next_tp_idx)
        # Preload next first (most likely direction), then prev
        !haskey(tp_data_cache, next_tp_idx) && push!(preload_tps, next_tp_idx)
        !haskey(tp_data_cache, prev_tp) && push!(preload_tps, prev_tp)
    end
    compare_mode[] && push!(needed_tps, compare_right_tp[])
    evict_tps = Int[k for k in keys(tp_data_cache) if !(k in needed_tps)]
    if !isempty(evict_tps) || !isempty(preload_tps)
        put!(io_channel[], EvictAndPreloadMessage(evict_tps, preload_tps))
    end
    println("  [BENCH] IO dispatched: evict=$(evict_tps), preload=$(preload_tps)"); flush(stdout)
    
    # Requirement: Automatically return to Lesion 1 when changing time points
    t_sync = @elapsed begin
        try
            reactToSyncLesion(SyncLesionEvent(1), stateObjects)
            println("Auto-reset to Lesion 1 for $label"); flush(stdout)
        catch e
            println("WARNING: Failed to auto-sync Lesion 1 on TP change: $e"); flush(stdout)
        end
    end
    println("  [BENCH] reactToSyncLesion(1): $(round(t_sync, digits=3))s"); flush(stdout)
    
    # Pre-compute SUV for all lesions in the current TP
    t_suv = @elapsed begin
        try
            LMW = Main.MedEye3d.LesionMetadataWindow
            if haskey(tp_data_cache, new_tp)
                cached_entry = tp_data_cache[new_tp]
                unique_ids = Set{Int}()
                for v in cached_entry.mask
                    iv = Int(v)
                    iv > 0 && push!(unique_ids, iv)
                end
                for lid in unique_ids
                    key = (new_tp, lid)
                    if !haskey(LMW._lesion_suv_cache, key)
                        try
                            LMW._lesion_suv_cache[key] = LMW.compute_lesion_suv_string(lid, new_tp)
                        catch; end
                    end
                end
                println("  [BENCH] SUV precomputed for $(length(unique_ids)) lesions"); flush(stdout)
            end
        catch e
            println("  [BENCH] SUV precompute skipped: $e"); flush(stdout)
        end
    end
    println("  [BENCH] SUV precompute: $(round(t_suv, digits=3))s"); flush(stdout)
    
    # Preload the new CT into Docker nnInteractive GPU memory (fire-and-forget)
    try
        if haskey(tp_data_cache, new_tp)
            cached_entry = tp_data_cache[new_tp]
            InferenceClient.preload_ct_for_nninteractive(Array{Float32,3}(cached_entry.ct))
            println("[TP Switch] CT preload initiated for $label"); flush(stdout)
        end
    catch e
        println("[TP Switch] CT preload skipped: $e"); flush(stdout)
    end
    
    t_total_ms = (time_ns() - t_total) / 1e6
    println("  [BENCH] TP SWITCH TOTAL: $(round(t_total_ms, digits=1))ms"); flush(stdout)
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
