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
export reactToChangeTimePoint, reactToSetTimePoint, reactToToggleLesion, reactToRefreshList
export reactToAddAutoPet, reactToAIInferenceResult, reactToSyncMissing, reactToGenManual
export reactToMapLink, reactToAutoRunPreprocess, reactToRunPreprocess, reactToShowBoneMask, reactToShowMaskLayer, reactToSaveMRB
export register_h5_mask_saver!, mark_tp_mask_dirty!, save_tp_mask_to_h5, flush_all_dirty_masks!, dirty_mask_tps
export _clinical_phase, get_clinical_phase, set_clinical_phase!, _phase_display_name
export reactToEditMode, reactToViewMode, reactToSetTPFirst, reactToSetTPLast, reactToToggleMaskVisibility, reactToShowOnlyPET, reactToShowOnlyCT, reactToToggleSyncScroll
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
const cursor_voxel_pos = Observable{Tuple{Int,Int,Int}}((0,0,0))
# Measurement info — dedicated Observable for live measurement values during drag
const measurement_info_text = Observable{String}("")  # "⬤ Sphere: SUVmean=0.42 SUVmax=1.23 R=10.0mm"

"""Safe Makie redraw trigger — uses a dedicated no-op Observable instead of
notify(fig.scene.visible) which can cause blinking by toggling scene visibility."""
function _safe_redraw(obs_dict::Dict{Symbol,Any})
    if haskey(obs_dict, :redraw_trigger)
        obs_dict[:redraw_trigger][] = obs_dict[:redraw_trigger][] + 1
    elseif haskey(obs_dict, :fig)
        # Fallback: notify px_area (always-present, no side effects)
        try notify(obs_dict[:fig].scene.px_area) catch; end
    end
end

using ...ScientificWorkflow
# --- Application State Objects ---
const _clinical_phase = Ref{ScientificWorkflow.ClinicalPhase}(ScientificWorkflow.PHASE_READ)
const _case_profile = Ref{ScientificWorkflow.CaseProfile}(ScientificWorkflow.PROFILE_GENERAL)
const _workflow_state = Ref{ScientificWorkflow.AnnotationWorkflowState}(ScientificWorkflow.WF_LESION_REVIEW)
const global_dicom_metadata = Ref{Dict{String,Any}}(Dict{String,Any}())

function get_workflow_state()
    return _workflow_state[]
end

function set_workflow_state!(state::ScientificWorkflow.AnnotationWorkflowState)
    old = _workflow_state[]
    if old != state
        _workflow_state[] = state
        @debug "[WORKFLOW] State changed: $old → $state"
    end
end

function get_clinical_phase()
    return _clinical_phase[]
end

function set_clinical_phase!(phase::ScientificWorkflow.ClinicalPhase)
    old = _clinical_phase[]
    if old != phase
        # Just update the ref for now, we'll avoid audit event since record_audit_event! might not be accessible here
        # or we could call it if it exists
        _clinical_phase[] = phase
        @debug "Clinical phase: $old -> $phase"
        # Update LMW display
        lmw = _get_lmw()
        if lmw !== nothing && haskey(lmw, :clinical_phase_obs)
            lmw[:clinical_phase_obs][] = _phase_display_name(phase)
        end
    end
end

function _phase_display_name(phase)
    phase == ScientificWorkflow.PHASE_CASE_SETUP && return "SETUP"
    phase == ScientificWorkflow.PHASE_READ && return "READ"
    phase == ScientificWorkflow.PHASE_ASSESS && return "ASSESS"
    phase == ScientificWorkflow.PHASE_REPORT_DRAFT && return "REPORT"
    phase == ScientificWorkflow.PHASE_VALIDATION && return "VALIDATE"
    phase == ScientificWorkflow.PHASE_SIGNED && return "SIGNED"
    return string(phase)
end

export reactToNextPhase, reactToPrevPhase, reactToSetPhase

function reactToNextPhase(data::NextPhaseEvent, stateObjects)
    current = get_clinical_phase()
    phases = [ScientificWorkflow.PHASE_CASE_SETUP, ScientificWorkflow.PHASE_READ, ScientificWorkflow.PHASE_ASSESS, ScientificWorkflow.PHASE_REPORT_DRAFT, ScientificWorkflow.PHASE_VALIDATION, ScientificWorkflow.PHASE_SIGNED]
    idx = findfirst(==(current), phases)
    if idx !== nothing && idx < length(phases)
        next_ph = phases[idx + 1]
        _run_phase_transition(current, next_ph, stateObjects)
    end
end

function reactToPrevPhase(data::PrevPhaseEvent, stateObjects)
    current = get_clinical_phase()
    phases = [ScientificWorkflow.PHASE_CASE_SETUP, ScientificWorkflow.PHASE_READ, ScientificWorkflow.PHASE_ASSESS, ScientificWorkflow.PHASE_REPORT_DRAFT, ScientificWorkflow.PHASE_VALIDATION, ScientificWorkflow.PHASE_SIGNED]
    idx = findfirst(==(current), phases)
    if idx !== nothing && idx > 1
        prev_ph = phases[idx - 1]
        _run_phase_transition(current, prev_ph, stateObjects)
    end
end

function reactToSetPhase(data::SetPhaseEvent, stateObjects)
    # mapping string to enum
    phases = [ScientificWorkflow.PHASE_CASE_SETUP, ScientificWorkflow.PHASE_READ, ScientificWorkflow.PHASE_ASSESS, ScientificWorkflow.PHASE_REPORT_DRAFT, ScientificWorkflow.PHASE_VALIDATION, ScientificWorkflow.PHASE_SIGNED]
    idx = findfirst(x -> _phase_display_name(x) == data.phase, phases)
    if idx !== nothing
        _run_phase_transition(get_clinical_phase(), phases[idx], stateObjects)
    end
end

function _run_phase_transition(old, new, stateObjects)
    if old == ScientificWorkflow.PHASE_READ && new == ScientificWorkflow.PHASE_ASSESS
        @info "Running basic validation (any critical unreviewed?)"
    elseif old == ScientificWorkflow.PHASE_ASSESS && new == ScientificWorkflow.PHASE_REPORT_DRAFT
        @info "Opening/refreshing E-PSMA report"
    elseif old == ScientificWorkflow.PHASE_REPORT_DRAFT && new == ScientificWorkflow.PHASE_VALIDATION
        @info "Running conflict checker"
    elseif old == ScientificWorkflow.PHASE_VALIDATION && new == ScientificWorkflow.PHASE_SIGNED
        @info "Locking case (no blocking issues)"
    end
    set_clinical_phase!(new)
end

const current_viewer_position = Ref((0, 0, 0))
const app_is_loading = Ref(true)
const current_hovered_panel = Ref{Int}(0)
export cursor_info_text, cursor_study_text, set_ai_status!, current_viewer_position, current_hovered_panel, measurement_info_text

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
    negative_coords::Vector{Vector{Int}}  # Negative points for nnInteractive
    spacing::Tuple{Float64,Float64,Float64}  # Real CT voxel spacing for correct AI model behavior
end

const inference_queue = Channel{InferenceJob}(8)

# Persistent worker thread — started once, processes jobs sequentially (no race conditions)
function start_inference_worker()
    Threads.@spawn begin
        @debug "[AI Worker] Inference worker thread started."
        while true
            try
                job = take!(inference_queue)
                if !InferenceClient.is_ai_enabled()
                    set_ai_status!("[AI Disabled] Restart app with AI enabled or run worker on port 5005")
                    continue
                end
                
                set_ai_status!("[Sending] to Docker ($(job.algorithm))...")
                @debug "[AI Worker] Processing $(job.algorithm) at ($(job.cx),$(job.cy),$(job.cz)) for lesion $(job.active_id)..."
                
                mask = nothing
                if job.algorithm == "NNInteractive"
                    # Fast path: use pre-extracted scribble coordinates (skip findall)
                    if !isempty(job.scribble_coords)
                        mask = InferenceClient.run_nninteractive(
                            job.ct_vol, job.pet_vol, job.scribble_coords, job.negative_coords,
                            job.cx, job.cy, job.cz; spacing=job.spacing)
                    else
                        mask = InferenceClient.run_nninteractive(
                            job.ct_vol, job.pet_vol, job.points_vol,
                            job.cx, job.cy, job.cz; spacing=job.spacing)
                    end
                elseif job.algorithm == "HELPNet (AI)"
                    mask = InferenceClient.run_helpnet_inference(
                        job.ct_vol, job.pet_vol, job.points_vol,
                        job.cx, job.cy, job.cz)
                else
                    @debug "[AI Worker] WARNING: Unknown algorithm: $(job.algorithm)"
                    set_ai_status!("[Warning] Unknown algorithm: $(job.algorithm)")
                    continue
                end
                
                if mask !== nothing
                    voxel_count = count(mask .> 0)
                    set_ai_status!("[Applying] result ($voxel_count voxels)...")
                    @debug "[AI Worker] Docker returned mask with $voxel_count voxels. Posting to channel."
                else
                    if !InferenceClient.is_worker_reachable()
                        err = InferenceClient.get_last_ai_error()
                        msg = isempty(err) ? "[Error] AI worker unreachable on port $(InferenceClient.get_ai_port()). Is Docker running?" : "[Error] AI offline: $err"
                        set_ai_status!(msg)
                        println("[AI Worker] $msg"); flush(stdout)
                    else
                        err = InferenceClient.get_last_ai_error()
                        msg = isempty(err) ? "[Warning] AI model returned no mask (inference returned empty)." : "[Warning] $err"
                        set_ai_status!(msg)
                        println("[AI Worker] $msg"); flush(stdout)
                    end
                end
                
                # Post result back to main event channel via on_next! multiple dispatch
                put!(job.main_channel, AIInferenceResultEvent(
                    job.algorithm, job.active_id,
                    job.cx, job.cy, job.cz,
                    mask, job.seg_vol))
                    
            catch e
                if e isa InvalidStateException  # channel closed
                    @debug "[AI Worker] Queue closed, shutting down."
                    break
                end
                err_msg = sprint(showerror, e)
                @debug "[AI Worker] ERROR: $err_msg"
                @debug "Error trace" exception=(e, catch_backtrace())
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
            wordsQuadVertSize = sizeof(_HIDDEN_QUAD_VERTS_W),
            imagePos = -1
        ))
    else
        pos = if layout == :TopLeft || layout == :LeftHalf || layout == :SingleImage || layout == :WholeWindow
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
        mode = (layout == :SingleImage || layout == :WholeWindow) ? SingleImage : ((layout == :LeftHalf || layout == :RightHalf) ? MultiImage : QuadImage)
        stateObject.displayMode = mode
        stateObject.calcDimsStruct = StructsManag.getMainVerticies(calcDimStruct, mode, pos)
    end
end

const compare_mode = Ref(false)
const compare_right_tp = Ref(-1)  # TP index shown in right panel (panel 5)
const tp_switched = Observable{Int}(0)
const _m2_reference_tp = Ref{Int}(-1)
const _m2_crosshair_sync = Ref(true)

const _flicker_active = Ref(false)
const _flicker_show_current = Ref(true)
const _flicker_timer = Ref{Union{Nothing,Timer}}(nothing)

# Measurement mode state
const measurements_mode = Observable{Bool}(false)
const active_measurement_radius_mm = Observable{Float32}(10.0f0)
const measurement_sub_mode = Observable{Symbol}(:sphere)  # :sphere or :line
export measurements_mode, active_measurement_radius_mm, measurement_sub_mode

"""Force direct texture upload for a panel — bypasses scroll pipeline entirely."""
function _force_texture_upload!(stateObjects::Vector{StateDataFields}, panel_idx::Int)
    panelState = stateObjects[panel_idx]
    dimToScroll = panelState.onScrollData.dimensionToScroll
    lastSlice = panelState.onScrollData.slicesNumber
    if lastSlice < 1
        println("  [FORCE-TEX] panel $panel_idx: slicesNumber=$lastSlice, SKIPPING upload"); flush(stdout)
        return
    end
    current = clamp(panelState.currentDisplayedSlice, 1, lastSlice)
    
    singleSlDat = panelState.onScrollData.dataToScroll |>
        (scrDat) -> map(threeDimDat -> begin
            td = threeToTwoDimm(threeDimDat.type, Int64(current), dimToScroll, threeDimDat)
            # Materialize SubArray/PermutedDimsArray views to contiguous arrays.
            # unsafe_copyto! in VulkanStaging requires contiguous memory;
            # selectdim() on PermutedDimsArray creates views with complex strides.
            materialized = td.dat isa Array ? td.dat : collect(td.dat)
            TwoDimRawDat{threeDimDat.type}(td.type, td.name, materialized)
        end, scrDat) |>
        (twoDimList) -> SingleSliceDat(listOfDataAndImageNames=twoDimList, sliceNumber=current, textToDisp=getTextForCurrentSlice(panelState.onScrollData, Int32(current)))
    
    # Safety: verify data dimensions fit within allocated texture
    n_textures = length(singleSlDat.listOfDataAndImageNames)
    for updateDat in singleSlDat.listOfDataAndImageNames
        actual_w = size(updateDat.dat, 1)
        actual_h = size(updateDat.dat, 2)
        tex_w = Int(panelState.calcDimsStruct.imageTextureWidth)
        tex_h = Int(panelState.calcDimsStruct.imageTextureHeight)
        if actual_w > tex_w || actual_h > tex_h
            println("  [FORCE-TEX] WARN: '$(updateDat.name)' on panel $panel_idx: data=$(actual_w)x$(actual_h) > texture=$(tex_w)x$(tex_h)"); flush(stdout)
        end
    end
    
    panelState.currentlyDispDat = singleSlDat
    panelState.currentDisplayedSlice = current
    panelState.isSliceChanged = true
    println("  [FORCE-TEX] panel $panel_idx: prepared $n_textures textures at slice $current (dimToScroll=$dimToScroll, slicesNumber=$lastSlice)"); flush(stdout)
end

function reactToToggleFlicker(data::MakieEvents.ToggleFlickerEvent, stateObjects::Vector{StateDataFields})
    _flicker_active[] = !_flicker_active[]
    if _flicker_active[]
        @info "Flicker mode ACTIVATED"
        # Ensure compare_right_tp is set
        if compare_right_tp[] < 0
            tp_indices = sort(collect(keys(tp_labels)))
            if !isempty(tp_indices)
                cur_pos = findfirst(==(current_tp_index[]), tp_indices)
                cur_pos = cur_pos === nothing ? 1 : cur_pos
if _m2_reference_tp[] == -1
    next_pos = cur_pos < length(tp_indices) ? cur_pos + 1 : 1
    right_tp = tp_indices[next_pos]
elseif _m2_reference_tp[] == 0
    right_tp = tp_indices[1]
else
    right_tp = _m2_reference_tp[]
end
                compare_right_tp[] = right_tp
            end
        end
        _flicker_show_current[] = true
        _flicker_timer[] = Timer(0.0, interval=0.5) do t
            _flicker_show_current[] = !_flicker_show_current[]
            tp_to_show = _flicker_show_current[] ? current_tp_index[] : compare_right_tp[]
            if tp_to_show >= 0
                entry = get_or_load_tp_data(tp_to_show)
                if entry !== nothing
                    for i in 6:9
                        _load_tp_from_entry!(stateObjects, entry, i)
                        _force_texture_upload!(stateObjects, i)
                    end
                end
            end
        end
    else
        @info "Flicker mode DEACTIVATED"
        if _flicker_timer[] !== nothing
            close(_flicker_timer[])
            _flicker_timer[] = nothing
        end
        entry = get_or_load_tp_data(current_tp_index[])
        if entry !== nothing
            for i in 6:9
                _load_tp_from_entry!(stateObjects, entry, i)
                _force_texture_upload!(stateObjects, i)
            end
        end
    end
end

function reactToToggleOverlay(data::MakieEvents.ToggleOverlayEvent, stateObjects::Vector{StateDataFields})
    @info "Overlay mode TOGGLED"
    # Same as apply_m2_layout! mode="Overlay"
    # But since it's an event, we'll just set the M2 layout string and apply it.
    # We can't access apply_m2_layout! directly from MakieEventHandlers unless we pass it or it's visible.
    # apply_m2_layout! is in SegmentationDisplay.jl which imports MakieEventHandlers.
end
function reactToCompareTimePoints(data::CompareTimePointsEvent, stateObjects::Vector{StateDataFields})
    if length(stateObjects) >= 5
        compare_mode[] = data.compare
        if data.compare
            println("[COMPARE] ON: preloading next TP data..."); flush(stdout)
            # Load the NEXT TP into panel 5 (hidden — for M2 window use)
            tp_indices = sort(collect(keys(tp_labels)))
            if !isempty(tp_indices)
                cur_pos = findfirst(==(current_tp_index[]), tp_indices)
                cur_pos = cur_pos === nothing ? 1 : cur_pos
if _m2_reference_tp[] == -1
    next_pos = cur_pos < length(tp_indices) ? cur_pos + 1 : 1
    right_tp = tp_indices[next_pos]
elseif _m2_reference_tp[] == 0
    right_tp = tp_indices[1]
else
    right_tp = _m2_reference_tp[]
end
                # right_tp assigned above
                compare_right_tp[] = right_tp
                println("[COMPARE] Loading right TP=$right_tp (current=$(current_tp_index[]))"); flush(stdout)
                
                # Load right TP data into panel 5
                entry = get_or_load_tp_data(right_tp)
                if entry !== nothing
                    println("[COMPARE] Got entry for TP=$right_tp, loading into panel 5..."); flush(stdout)
                    _load_tp_from_entry!(stateObjects, entry, 5)
                    println("[COMPARE] Panel 5 data loaded"); flush(stdout)
                else
                    println("[COMPARE] WARNING: get_or_load_tp_data returned nothing for TP=$right_tp"); flush(stdout)
                end
            end

            # Sync panel 5 scroll/zoom with panel 1 (for M2 use)
            stateObjects[5].onScrollData.dimensionToScroll = stateObjects[1].onScrollData.dimensionToScroll
            stateObjects[5].currentDisplayedSlice = stateObjects[1].currentDisplayedSlice
            stateObjects[5].onScrollData.slicesNumber = stateObjects[1].onScrollData.slicesNumber
            stateObjects[5].calcDimsStruct.zoom = stateObjects[1].calcDimsStruct.zoom
            stateObjects[5].calcDimsStruct.panX = stateObjects[1].calcDimsStruct.panX
            stateObjects[5].calcDimsStruct.panY = stateObjects[1].calcDimsStruct.panY
            stateObjects[5].calcDimsStruct.imageTextureWidth = stateObjects[1].calcDimsStruct.imageTextureWidth
            stateObjects[5].calcDimsStruct.imageTextureHeight = stateObjects[1].calcDimsStruct.imageTextureHeight
            stateObjects[5].calcDimsStruct.heightToWithRatio = stateObjects[1].calcDimsStruct.heightToWithRatio

            # Keep quad layout — NO vertex changes. Panel 5 stays hidden.
            # The actual side-by-side comparison happens in the M2 window.
            updateQuadVertices!(stateObjects[5], :Hidden)

            left_label = get(tp_labels, current_tp_index[], "TP $(current_tp_index[])")
            right_label = get(tp_labels, compare_right_tp[], "TP $(compare_right_tp[])")
            println("[COMPARE] Current=$left_label, Compare=$right_label (data preloaded in panel 5)"); flush(stdout)
            
            # Preload panel 5 texture data (ready for M2)
            _force_texture_upload!(stateObjects, 5)
            
            # Sync active lesion visibility
            if current_active_lesion_id[] > 0
                try
                    reactToSyncLesion(SyncLesionEvent(current_active_lesion_id[]), stateObjects)
                catch e
                    println("[COMPARE] WARNING: reactToSyncLesion failed: $e"); flush(stdout)
                end
            end
            println("[COMPARE] ON: setup complete (quad view unchanged, data preloaded for M2)"); flush(stdout)
        else
            println("[COMPARE] OFF: clearing compare state"); flush(stdout)
            compare_right_tp[] = -1
            # Hide panel 5 (compare data no longer needed)
            updateQuadVertices!(stateObjects[5], :Hidden)
            
            # Quad layout was never changed, so no vertex restoration needed.
            # Just sync lesion visibility back to single-TP mode.
            if current_active_lesion_id[] > 0
                try
                    lid_off = _clamp_lid_for_tp(current_active_lesion_id[], current_tp_index[])
                    reactToSyncLesion(SyncLesionEvent(lid_off), stateObjects)
                catch e
                    @debug "WARNING: reactToSyncLesion failed during compare-OFF: $e"
                end
            end
            println("[COMPARE] OFF: done"); flush(stdout)
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
                    textSpec.minAndMaxValue = T_mm.([1, 10000])
                else
                    textSpec.minAndMaxValue = T_mm.([data.lesion_id, data.lesion_id])
                end
                changed = true
            end
        end
    end
    lbl = is_single_lesion_mode[] ? string(data.lesion_id) : "all"
    @debug "Show single lesion: $lbl (single_mode=$(is_single_lesion_mode[]))"
    _mri_clamp_mask_range!(stateObjects)
    return changed
end

const current_windowing = Dict{String, Vector{Float32}}(
    "CT"    => Float32[-160.0, 240.0],
    "PET"   => Float32[0.0, 10.0],
    "SPECT" => Float32[0.0, 10.0],
    "T2"    => Float32[0.0, 1000.0],
    "MRI"   => Float32[0.0, 1000.0],
    "MR"    => Float32[0.0, 1000.0],
    "T1"    => Float32[0.0, 600.0],
    "ADC"   => Float32[0.0, 1500.0],
    "DWI"   => Float32[0.0, 500.0]
)
export current_windowing

function reactToWindowing(data::WindowingEvent, stateObjects::Vector{StateDataFields})
    target_mod = uppercase(data.modality)
    current_windowing[target_mod] = Float32.([data.min_val, data.max_val])
    if target_mod in ("T2", "MRI", "MR")
        current_windowing["T2"] = Float32.([data.min_val, data.max_val])
        current_windowing["MRI"] = Float32.([data.min_val, data.max_val])
        current_windowing["MR"] = Float32.([data.min_val, data.max_val])
    end
    
    for (panel_idx, state) in enumerate(stateObjects)
        panel_tp = if compare_mode[] && panel_idx == 5
            compare_right_tp[]
        else
            state.onScrollData.currentTpIndex > 0 ? state.onScrollData.currentTpIndex : current_tp_index[]
        end
        panel_mod = uppercase(get(tp_modalities, panel_tp, "PET"))
        
        is_main_match = false
        is_nuc_match = false
        
        if target_mod == "CT"
            is_main_match = (panel_mod == "CT" || panel_mod == "PET" || panel_mod == "SPECT")
        elseif target_mod in ("T2", "MRI", "MR")
            is_main_match = (panel_mod in ("T2", "MRI", "MR"))
        elseif target_mod == "T1"
            is_main_match = (panel_mod == "T1")
        elseif target_mod == "ADC"
            is_main_match = (panel_mod == "ADC")
            is_nuc_match = (panel_mod in ("T2", "MRI", "MR") || panel_mod == "ADC")
        elseif target_mod == "DWI"
            is_main_match = (panel_mod == "DWI")
            is_nuc_match = (panel_mod == "DWI")
        elseif target_mod == "PET"
            is_nuc_match = (panel_mod == "PET" || panel_mod == "CT")
        elseif target_mod == "SPECT"
            is_nuc_match = (panel_mod == "SPECT")
        end
        
        for tex in state.mainForDisplayObjects.listOfTextSpecifications
            # Match CT textures: studyType=="CT", or fallback to isMainImage when studyType is unset
            # (but exclude PET/SPECT textures that have isMainImage=true on pure-PET panels)
            is_ct_tex = tex.studyType == "CT" || (isempty(tex.studyType) && tex.isMainImage && !tex.isNuclearMask && !(uppercase(tex.name) in ("PET", "SPECT")))
            is_nuc_tex = tex.studyType == "PET" || tex.studyType == "SPECT" || tex.isNuclearMask
            if is_main_match && is_ct_tex
                tex.minAndMaxValue = Float32.([data.min_val, data.max_val])
            elseif is_nuc_match && is_nuc_tex
                tex.minAndMaxValue = Float32.([data.min_val, data.max_val])
            end
        end
    end
    @debug "Updated windowing for $(data.modality): [$(data.min_val), $(data.max_val)]"
end

function reactToPetBlend(data::PetBlendEvent, stateObjects::Vector{StateDataFields})
    range_to_update = data.window_id == 0 ? (1:length(stateObjects)) : (data.window_id == 2 ? (6:10) : (1:5))
    for idx in range_to_update
        if idx > length(stateObjects)
            continue
        end
        state = stateObjects[idx]
        for tex in state.mainForDisplayObjects.listOfTextSpecifications
            if tex.isNuclearMask && !tex.isMainImage
                tex.maskContribution = clamp(data.weight, 0.0f0, 1.0f0)
            end
        end
        if state.mainForDisplayObjects.vulkanPipelineState !== nothing
            state.mainForDisplayObjects.vulkanPipelineState.ubo_dirty = true
        end
    end
    @debug "PET/CT blend updated" weight=data.weight window_id=data.window_id
    # Sync GUI blend slider (with guard to prevent circular event loop)
    try
        LMW = _get_lmw()
        if LMW !== nothing
            obs_dict = getfield(LMW, :_lmw_observables)
            if haskey(obs_dict, :slider_blend)
                if haskey(obs_dict, :is_syncing_blend)
                    obs_dict[:is_syncing_blend][] = true
                end
                try
                    obs_dict[:slider_blend].value[] = data.weight
                finally
                    if haskey(obs_dict, :is_syncing_blend)
                        obs_dict[:is_syncing_blend][] = false
                    end
                end
            end
            _safe_redraw(obs_dict)
        end
    catch; end
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
    @debug "Paint state updated: val=$(data.val), active=$(data.active)"
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

# ── EditModeEvent (E key) ──────────────────────────────────────────────────
# Same as clicking the "Paint" button in the Makie metadata window.
# Activates paint mode with the current lesion ID, updates GUI button colors.
function reactToEditMode(data::MakieEvents.EditModeEvent, stateObjects::Vector{StateDataFields})
    lid = current_active_lesion_id[] > 0 ? current_active_lesion_id[] : 1
    # Activate painting in render state (same as PaintValEvent)
    reactToPaintVal(PaintValEvent(lid, true), stateObjects)
    # Update GUI: paint button highlighted, workflow state
    try
        LMW = _get_lmw()
        if LMW !== nothing
            obs_dict = getfield(LMW, :_lmw_observables)
            if haskey(obs_dict, :obs_edit_mode)
                obs_dict[:obs_edit_mode][] = obs_dict[:obs_edit_mode][] + 1
            end
        end
    catch; end
    set_workflow_state!(ScientificWorkflow.WF_EDIT_MASK)
    @info "[KEYBOARD] E → Edit/Paint mode activated (lesion $lid)"
end

# ── ViewModeEvent (Esc key) ────────────────────────────────────────────────
# Same as clicking the "View" button in the Makie metadata window.
# Deactivates paint, returns to view mode, updates GUI button colors.
function reactToViewMode(data::MakieEvents.ViewModeEvent, stateObjects::Vector{StateDataFields})
    # Deactivate painting in render state (same as PaintValEvent(-1, false))
    reactToPaintVal(PaintValEvent(-1, false), stateObjects)
    # Update GUI: view button highlighted, workflow state
    try
        LMW = _get_lmw()
        if LMW !== nothing
            obs_dict = getfield(LMW, :_lmw_observables)
            if haskey(obs_dict, :obs_view_mode)
                obs_dict[:obs_view_mode][] = obs_dict[:obs_view_mode][] + 1
            end
        end
    catch; end
    set_workflow_state!(ScientificWorkflow.WF_LESION_REVIEW)
    @info "[KEYBOARD] Esc → View mode activated"
end

# ── SetTPFirstEvent (Home key) ─────────────────────────────────────────────
# Jump to first TP (baseline). Same as selecting first option in TP dropdown.
function reactToSetTPFirst(data::MakieEvents.SetTPFirstEvent, stateObjects::Vector{StateDataFields})
    if isempty(tp_labels)
        @debug "No TP labels loaded. Home key ignored."
        return
    end
    tp_indices = sort(collect(keys(tp_labels)))
    first_tp = tp_indices[1]
    if current_tp_index[] != first_tp
        reactToSetTimePoint(SetTimePointEvent(first_tp), stateObjects)
        @info "[KEYBOARD] Home → Jump to first TP (index=$first_tp)"
    end
end

# ── SetTPLastEvent (End key) ───────────────────────────────────────────────
# Jump to last TP. Same as selecting last option in TP dropdown.
function reactToSetTPLast(data::MakieEvents.SetTPLastEvent, stateObjects::Vector{StateDataFields})
    if isempty(tp_labels)
        @debug "No TP labels loaded. End key ignored."
        return
    end
    tp_indices = sort(collect(keys(tp_labels)))
    last_tp = tp_indices[end]
    if current_tp_index[] != last_tp
        reactToSetTimePoint(SetTimePointEvent(last_tp), stateObjects)
        @info "[KEYBOARD] End → Jump to last TP (index=$last_tp)"
    end
end

const _shortcut_visibility_cache = Dict{UInt64, Bool}()

# ── ToggleMaskVisibilityEvent (Q hold/release) ────────────────────────────
# Q press: hide all masks. Q release: restore masks.
function reactToToggleMaskVisibility(data::MakieEvents.ToggleMaskVisibilityEvent, stateObjects::Vector{StateDataFields})
    is_press = !data.visible  # true = show (release), false = hide (press)
    for state in stateObjects
        for textSpec in state.mainForDisplayObjects.listOfTextSpecifications
            if is_press
                if !haskey(_shortcut_visibility_cache, objectid(textSpec))
                    _shortcut_visibility_cache[objectid(textSpec)] = textSpec.isVisible
                end
                if textSpec.name == "Mask" || (textSpec.isMultiDiscreteMask && textSpec.name != "Anatomy")
                    textSpec.isVisible = false
                end
            else
                if haskey(_shortcut_visibility_cache, objectid(textSpec))
                    textSpec.isVisible = _shortcut_visibility_cache[objectid(textSpec)]
                end
            end
        end
        state.isSliceChanged = true
    end
    if !is_press
        empty!(_shortcut_visibility_cache)
    end
    @debug "[KEYBOARD] Q → Mask visibility: $(data.visible ? "restored" : "hidden")"
end

# ── ShowOnlyPETEvent (P hold) ──────────────────────────────────────────────
# P hold: show only main images and PET/SPECT, hide all masks/overlays
function reactToShowOnlyPET(data::MakieEvents.ShowOnlyPETEvent, stateObjects::Vector{StateDataFields})
    for state in stateObjects
        for textSpec in state.mainForDisplayObjects.listOfTextSpecifications
            if data.active  # press: hide all masks except PET/SPECT
                if !haskey(_shortcut_visibility_cache, objectid(textSpec))
                    _shortcut_visibility_cache[objectid(textSpec)] = textSpec.isVisible
                end
                is_ct_tex = textSpec.studyType == "CT" || (isempty(textSpec.studyType) && textSpec.isMainImage && !textSpec.isNuclearMask && !(uppercase(textSpec.name) in ("PET", "SPECT")))
                is_pet_spect = textSpec.studyType == "PET" || textSpec.studyType == "SPECT" || textSpec.isNuclearMask
                
                if is_ct_tex || is_pet_spect || textSpec.isMainImage
                    textSpec.isVisible = true
                else
                    textSpec.isVisible = false
                end
            else  # release: restore all
                if haskey(_shortcut_visibility_cache, objectid(textSpec))
                    textSpec.isVisible = _shortcut_visibility_cache[objectid(textSpec)]
                end
            end
        end
        state.isSliceChanged = true
    end
    if !data.active
        empty!(_shortcut_visibility_cache)
    end
end

# ── ShowOnlyCTEvent (T hold) ───────────────────────────────────────────────
# T hold: show only CT textures  
function reactToShowOnlyCT(data::MakieEvents.ShowOnlyCTEvent, stateObjects::Vector{StateDataFields})
    for state in stateObjects
        for textSpec in state.mainForDisplayObjects.listOfTextSpecifications
            if data.active  # press: hide everything except CT
                if !haskey(_shortcut_visibility_cache, objectid(textSpec))
                    _shortcut_visibility_cache[objectid(textSpec)] = textSpec.isVisible
                end
                is_ct_tex = textSpec.studyType == "CT" || (isempty(textSpec.studyType) && textSpec.isMainImage && !textSpec.isNuclearMask && !(uppercase(textSpec.name) in ("PET", "SPECT")))
                
                if is_ct_tex
                    textSpec.isVisible = true
                else
                    textSpec.isVisible = false
                end
            else  # release: restore all
                if haskey(_shortcut_visibility_cache, objectid(textSpec))
                    textSpec.isVisible = _shortcut_visibility_cache[objectid(textSpec)]
                end
            end
        end
        state.isSliceChanged = true
    end
    if !data.active
        empty!(_shortcut_visibility_cache)
    end
end

# ── ToggleSyncScrollEvent (S key) ──────────────────────────────────────────
# S key: toggle synchronized scrolling across panels + sync GUI button
function reactToToggleSyncScroll(data::MakieEvents.ToggleSyncScrollEvent, stateObjects::Vector{StateDataFields})
    for state in stateObjects
        state.mainForDisplayObjects.isSyncScrollOn = !state.mainForDisplayObjects.isSyncScrollOn
    end
    # Sync GUI button
    try
        LMW = _get_lmw()
        if LMW !== nothing
            obs_dict = getfield(LMW, :_lmw_observables)
            if haskey(obs_dict, :obs_sync_scroll_changed)
                obs_dict[:obs_sync_scroll_changed][] = obs_dict[:obs_sync_scroll_changed][] + 1
            end
        end
    catch; end
end

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
    
    # === Erosion-based bone surface extraction ===
    # Compute 6-connected erosion: a bone voxel is "interior" only if ALL 6 face-adjacent
    # neighbors are also bone. Surface = bone & ~interior (1-voxel-thick shell).
    cx, cy, cz = size(crop_skelly)
    crop_bone_bool = Array{Bool}(undef, cx, cy, cz)
    @inbounds for i in eachindex(crop_skelly)
        crop_bone_bool[i] = crop_skelly[i] > 0
    end
    
    is_interior = falses(cx, cy, cz)
    @inbounds for k in 2:cz-1, j in 2:cy-1, i in 2:cx-1
        if crop_bone_bool[i,j,k] &&
           crop_bone_bool[i-1,j,k] && crop_bone_bool[i+1,j,k] &&
           crop_bone_bool[i,j-1,k] && crop_bone_bool[i,j+1,k] &&
           crop_bone_bool[i,j,k-1] && crop_bone_bool[i,j,k+1]
            is_interior[i,j,k] = true
        end
    end
    # Bone surface = bone voxels that have at least one non-bone neighbor
    # (boundary voxels at crop edges are always surface by definition)
    
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
        
        # Surface: only bone voxels on the outer shell (NOT interior) within distance
        if !is_interior[b_idx] && min_d2 <= max_surf_dist_mm2
            push!(surf_pts, global_idx)
        end
        
        # Marrow: interior bone voxels within distance, excluding the lesion itself
        if is_interior[b_idx] && min_d2 <= max_marr_dist_mm2 && crop_mask[b_idx] != target_id
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
    mask_vol = lock(_tp_cache_lock) do
        haskey(tp_data_cache, panel_tp) ? tp_data_cache[panel_tp].mask_i16 : nothing
    end
    if mask_vol === nothing
        if stateObject !== nothing && isdefined(stateObject, :onScrollData)
            for scr in stateObject.onScrollData.dataToScroll
                if scr.name == "Mask"
                    mask_vol = scr.dat
                    break
                end
            end
        end
    end
    
    # Cache bone atlas presence check (avoid scanning entire volume every time)
    has_bone = if _bone_atlas_has_data[] !== nothing
        _bone_atlas_has_data[]
    elseif skelly_vol !== nothing
        v = count(skelly_vol .> 0) > 0
        _bone_atlas_has_data[] = v
        v
    else
        false
    end
    
    if skelly_vol !== nothing && mask_vol !== nothing && has_bone
        # Get spacing from stateObject or use default
        sp = try
            sv = stateObject.spacingsValue
            isa(sv, Tuple) ? sv : sv[1]
        catch
            (1.0, 1.0, 2.0)
        end
        
        # Crop to bounding box for transfer efficiency
        sz = size(mask_vol)
        lesion_vox = findall(mask_vol .== target_id)
        if isempty(lesion_vox)
            bone_subsegments_cache[(panel_tp, target_id)] = (CartesianIndex{3}[], CartesianIndex{3}[])
            return (CartesianIndex{3}[], CartesianIndex{3}[])
        end
        xs = [I[1] for I in lesion_vox]; ys = [I[2] for I in lesion_vox]; zs = [I[3] for I in lesion_vox]
        margin = 30
        x_min = max(1, minimum(xs) - margin); x_max = min(sz[1], maximum(xs) + margin)
        y_min = max(1, minimum(ys) - margin); y_max = min(sz[2], maximum(ys) + margin)
        z_min = max(1, minimum(zs) - margin); z_max = min(sz[3], maximum(zs) + margin)
        
        crop_mask = view(mask_vol, x_min:x_max, y_min:y_max, z_min:z_max)
        crop_skelly = view(skelly_vol, x_min:x_max, y_min:y_max, z_min:z_max)
        
        les_arr = convert(Array{UInt8,3}, crop_mask .== target_id)
        bone_arr = convert(Array{UInt8,3}, crop_skelly .> 0)
        
        # Mark as computing (sentinel prevents re-dispatch — see line 808)
        bone_subsegments_cache[(panel_tp, target_id)] = :computing
        if !isempty(node_name)
            bone_subsegments_cache[(node_name, target_id)] = :computing
        end
        
        # Async: spawn background task for the remote call so consumer is NOT blocked
        let les_arr=les_arr, bone_arr=bone_arr, sp=sp,
            x_min=x_min, y_min=y_min, z_min=z_min,
            panel_tp=panel_tp, target_id=target_id, node_name=node_name,
            mask_vol=mask_vol, skelly_vol=skelly_vol
            Threads.@spawn begin
                try
                    println("  [BONE-ASYNC] Starting remote bone subseg for lid=$target_id tp=$panel_tp"); flush(stdout)
                    surf_crop, marr_crop = Main.MedEye3d.InferenceClient.run_bone_subsegmentation_remote(les_arr, bone_arr, sp)
                    
                    s_pts = CartesianIndex{3}[]
                    m_pts = CartesianIndex{3}[]
                    if surf_crop !== nothing && marr_crop !== nothing
                        for I in findall(surf_crop)
                            push!(s_pts, CartesianIndex(I[1] + x_min - 1, I[2] + y_min - 1, I[3] + z_min - 1))
                        end
                        for I in findall(marr_crop)
                            push!(m_pts, CartesianIndex(I[1] + x_min - 1, I[2] + y_min - 1, I[3] + z_min - 1))
                        end
                    end
                    
                    println("  [BONE-ASYNC] Remote done: $(length(s_pts)) surf, $(length(m_pts)) marrow"); flush(stdout)
                    
                    # Feed result back to consumer via existing BoneSubsegResultEvent
                    ch = main_event_channel[]
                    if ch !== nothing
                        put!(ch, BoneSubsegResultEvent(panel_tp, target_id, s_pts, m_pts))
                    end
                catch e
                    println("  [BONE-ASYNC] Remote failed: $e — trying fast fallback"); flush(stdout)
                    try
                        res = compute_bone_subsegments_fast(mask_vol, skelly_vol, target_id)
                        ch = main_event_channel[]
                        if ch !== nothing
                            put!(ch, BoneSubsegResultEvent(panel_tp, target_id, res[1], res[2]))
                        end
                    catch e2
                        println("  [BONE-ASYNC] Fast fallback also failed: $e2"); flush(stdout)
                        bone_subsegments_cache[(panel_tp, target_id)] = (CartesianIndex{3}[], CartesianIndex{3}[])
                        if !isempty(node_name)
                            bone_subsegments_cache[(node_name, target_id)] = (CartesianIndex{3}[], CartesianIndex{3}[])
                        end
                    end
                end
            end
        end
        
        # Return empty immediately — overlay will be populated when BoneSubsegResultEvent arrives
        return (CartesianIndex{3}[], CartesianIndex{3}[])
    end
    
    bone_subsegments_cache[(panel_tp, target_id)] = (CartesianIndex{3}[], CartesianIndex{3}[])
    return (CartesianIndex{3}[], CartesianIndex{3}[])
end

function reactToSyncLesion(data::SyncLesionEvent, stateObjects::Vector{StateDataFields})
    println("[SYNC-LESION] START lid=$(data.lesion_id)"); flush(stdout)
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
            @debug "WARNING: Error finding cross-TP lesion: $e"
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
                        textSpec.minAndMaxValue = T_mm.([1, 10000])
                    end
                end
            elseif textSpec.name == "manualModif"
                textSpec.minAndMaxValue = Float32.([0.0, 10000.0])
                textSpec.allowedIDs = Float32[]
            end
        end
    end
    _mri_clamp_mask_range!(stateObjects)

    println("[SYNC-LESION] Visibility updated, bone subseg..."); flush(stdout)
    # 1b. Update bone subseg 3D arrays for visible panels only
    has_any_bone_data = false
    if data.lesion_id > 0
        for (panel_idx, stateObject) in enumerate(stateObjects)
            # Skip hidden panels (Fix ❸: avoid bone overlay work for invisible panels)
            if stateObject.calcDimsStruct.mainQuadVertSize <= 0 || all(iszero, stateObject.calcDimsStruct.mainImageQuadVert)
                continue
            end
            is_right_tp = compare_mode[] && panel_idx in (5, 6, 7, 8, 9)
            panel_tp = is_right_tp ? compare_right_tp[] : current_tp_index[]
            panel_lid = is_right_tp ? panel5_lesion_id : data.lesion_id

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
                    # Clear previous overlay (with bounds filtering)
                    if haskey(last_bone_overlay_indices, panel_idx) && !isempty(last_bone_overlay_indices[panel_idx])
                        valid_old = filter(idx -> checkbounds(Bool, scrDat.dat, idx), last_bone_overlay_indices[panel_idx])
                        if !isempty(valid_old)
                            scrDat.dat[valid_old] .= Int8(0)
                        end
                    end
                    # Write combined mask: surface=1, marrow=2, both=3
                    all_indices = CartesianIndex{3}[]
                    if !isempty(surf_indices)
                        valid_surf = filter(idx -> checkbounds(Bool, scrDat.dat, idx), surf_indices)
                        if !isempty(valid_surf)
                            scrDat.dat[valid_surf] .= Int8(1)
                        end
                        append!(all_indices, valid_surf)
                    end
                    if !isempty(marr_indices)
                        # For overlapping voxels, add (1+2=3), for marrow-only set to 2
                        for idx in marr_indices
                            if checkbounds(Bool, scrDat.dat, idx)
                                old_val = scrDat.dat[idx]
                                scrDat.dat[idx] = old_val == Int8(1) ? Int8(3) : Int8(2)
                            end
                        end
                        append!(all_indices, filter(idx -> checkbounds(Bool, scrDat.dat, idx), marr_indices))
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

    println("[SYNC-LESION] Bone subseg dispatched, centroid..."); flush(stdout)
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
            if i in (1, 2, 5, 6, 7)  # Axial panels (main + M2)
                stateObjects[i].lastRecordedMousePosition = CartesianIndex(origX, origY, origZ)
            elseif i in (3, 8)  # Sagittal panels (main + M2)
                stateObjects[i].lastRecordedMousePosition = CartesianIndex(origY, origZ, origX)
            elseif i in (4, 9)  # Coronal panels (main + M2)
                stateObjects[i].lastRecordedMousePosition = CartesianIndex(origX, origZ, origY)
            end
        end

        targets = [(1, origZ), (2, origZ), (3, origX), (4, origY)]
        if length(stateObjects) >= 5
            push!(targets, (5, origZ))
        end
        # M2 Quad View panels mirror main window layout
        if length(stateObjects) >= 9
            push!(targets, (6, origZ))  # M2 Axial (like panel 1)
            push!(targets, (7, origZ))  # M2 PET (like panel 2)
            push!(targets, (8, origX))  # M2 Sagittal (like panel 3)
            push!(targets, (9, origY))  # M2 Coronal (like panel 4)
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
    println("[SYNC-LESION] DONE $(round(t_total_ms, digits=1))ms"); flush(stdout)
    return changed
end

# TP navigation state: compact cache holding only base axial volumes
const _bone_atlas_has_data = Ref{Union{Nothing, Bool}}(nothing)
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
const _tp_cache_lock = ReentrantLock()   # Protects tp_data_cache Dict operations (short-lived)
const _hdf5_io_lock = ReentrantLock()    # Serializes ALL HDF5 file access (libhdf5 is NOT thread-safe)
const bone_subsegments_cache = Dict{Any, Any}()
const lesion_centroids_cache = Dict{Any, Vector{Int}}()
const _centroids_lock = ReentrantLock()
const last_bone_overlay_indices = Dict{Int, Vector{CartesianIndex{3}}}()
function reactToBoneSubsegResult(data::BoneSubsegResultEvent, stateObjects::Vector{StateDataFields})
    println("[BONE-RESULT] Received: lid=$(data.target_id) tp=$(data.panel_tp) surf=$(length(data.pts_surf)) marr=$(length(data.pts_marr))"); flush(stdout)
    bone_subsegments_cache[(data.panel_tp, data.target_id)] = (data.pts_surf, data.pts_marr)
    
    # Re-render if this lesion is still the active one
    if current_active_lesion_id[] == data.target_id && data.target_id > 0
        reactToSyncLesion(SyncLesionEvent(data.target_id), stateObjects)
    end
end

"""Clamp lesion_id to the range of segments available on the given TP.
Falls back to 1 if the current ID doesn't exist on the target TP."""
function _clamp_lid_for_tp(lid::Int, tp::Int)::Int
    if lid <= 0; return 1; end
    # Check tp_segment_names (populated from scene_hierarchy / segment_names.json)
    if haskey(tp_segment_names, tp)
        seg_ids = collect(keys(tp_segment_names[tp]))
        if !isempty(seg_ids) && !(lid in seg_ids)
            return minimum(seg_ids)
        end
    end
    # Fallback: check mask data cache for max segment ID
    max_id = lock(_tp_cache_lock) do
        haskey(tp_data_cache, tp) ? Int(maximum(tp_data_cache[tp].mask)) : 0
    end
    if max_id > 0 && lid > max_id
        return 1
    end
    return lid
end

"""On MRI modalities, show lesion segments in Mask texture.
If anatomy toggle is ON, also show prostate gland (label 4+).
Call this after reactToSyncLesion which may have re-applied single-lesion filtering."""
function _force_mri_show_all!(stateObjects::Vector{StateDataFields})
    max_label = 3  # MRI: always clamp mask to lesion labels 1-3 (never show gland label 4)

    for (i, stateObject) in enumerate(stateObjects)
        tp = (i == 5 && compare_mode[]) ? compare_right_tp[] : current_tp_index[]
        panel_mod = uppercase(get(tp_modalities, tp, "PET"))
        if !(panel_mod in ("T2", "MRI", "MR", "T1", "ADC", "DWI"))
            continue
        end
        for textSpec in stateObject.mainForDisplayObjects.listOfTextSpecifications
            if textSpec.name == "Mask" || textSpec.name == "segmentation" || (textSpec.isMultiDiscreteMask && textSpec.name != "Anatomy" && textSpec.name != "Bone_Overlay")
                T_mm = eltype(textSpec.minAndMaxValue)
                textSpec.minAndMaxValue = T_mm.([1, max_label])
                textSpec.isVisible = true
                # Ensure mask is rendered with non-zero opacity
                if textSpec.maskContribution <= 0.0f0
                    textSpec.maskContribution = 0.5f0
                end
            end
        end
    end
end

"""Check if timepoint modality has PET/SPECT data (not MRI-only)."""
function _has_nuclear_modality(tp::Int)::Bool
    panel_mod = uppercase(get(tp_modalities, tp, "PET"))
    return !(panel_mod in ("T2", "MRI", "MR", "T1", "ADC", "DWI"))
end

"""Clamp mask minAndMaxValue on MRI TPs to hide gland (label 4+).
On MRI, mask is always clamped to [1,3] — anatomy toggle controls the separate Anatomy texture.
Safe to call on any modality — no-ops on non-MRI TPs."""
function _mri_clamp_mask_range!(stateObjects::Vector{StateDataFields})

    # Check each panel individually based on its actual TP
    for (i, stateObject) in enumerate(stateObjects)
        tp = (i == 5 && compare_mode[]) ? compare_right_tp[] : current_tp_index[]
        panel_mod = uppercase(get(tp_modalities, tp, "PET"))
        if !(panel_mod in ("T2", "MRI", "MR", "T1", "ADC", "DWI"))
            continue
        end
        for textSpec in stateObject.mainForDisplayObjects.listOfTextSpecifications
            if textSpec.name == "Mask" || textSpec.name == "segmentation" || (textSpec.isMultiDiscreteMask && textSpec.name != "Anatomy" && textSpec.name != "Bone_Overlay")
                T_mm = eltype(textSpec.minAndMaxValue)
                cur_max = textSpec.minAndMaxValue[2]
                if cur_max > T_mm(3)
                    textSpec.minAndMaxValue = T_mm.([textSpec.minAndMaxValue[1], 3])
                end
            end
        end
    end
end

"""Update quad layout: hide Panel 2 (PET-only) when modality is MRI."""
function _update_quad_layout_for_modality!(stateObjects::Vector{StateDataFields}, tp::Int)
    if _has_nuclear_modality(tp)
        updateQuadVertices!(stateObjects[2], :TopRight)
    else
        updateQuadVertices!(stateObjects[2], :Hidden)
        @debug "[LAYOUT] Panel 2 (PET-only) hidden — MRI modality for TP $tp"
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
    # Load measurements now that HDF5 path is available
    try
        if _main_obj_ref[] !== nothing
            println("  [MEAS-INIT] Deferred load: tp=$(current_tp_index[]), h5=$h5_path"); flush(stdout)
            load_measurements_from_h5!(current_tp_index[], _main_obj_ref[])
        else
            # _main_obj_ref not yet set — schedule a deferred load
            @async begin
                for _wait_i in 1:50  # Wait up to 5 seconds
                    sleep(0.1)
                    if _main_obj_ref[] !== nothing
                        println("  [MEAS-INIT] Deferred load (async, after $(_wait_i*100)ms): tp=$(current_tp_index[])"); flush(stdout)
                        load_measurements_from_h5!(current_tp_index[], _main_obj_ref[])
                        break
                    end
                end
            end
        end
    catch e
        println("  [MEAS-INIT] Deferred load error: $e"); flush(stdout)
    end
end

function mark_tp_mask_dirty!(tp_idx::Int)
    lock(_mask_save_lock) do
        push!(dirty_mask_tps, tp_idx)
    end
    try
        LMW = _get_lmw()
        if LMW !== nothing && isdefined(LMW, :_cached_lesion_ids)
            delete!(LMW._cached_lesion_ids, tp_idx)
        end
    catch; end
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
    
    entry = lock(_tp_cache_lock) do
        haskey(tp_data_cache, tp_i) ? tp_data_cache[tp_i] : nothing
    end
    if entry === nothing
        return false
    end
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
            lock(_hdf5_io_lock) do
                HDF5.h5open(h5_path, "r+") do h5_file
                    is_pf = haskey(h5_file, "_meta_/preflipped") && read(h5_file["_meta_/preflipped"]) == 1
                    needs_reverse = !is_pf
                    
                    raw_to_write = needs_reverse ? reverse(mask_to_save, dims=2) : mask_to_save
                    ds_path_expert = "$group/$(mask_fname)_expert"
                    if haskey(h5_file, ds_path_expert)
                        h5_file[ds_path_expert][:, :, :] = Int16.(raw_to_write)
                    else
                        h5_file[ds_path_expert, chunk=(32,32,32), compress=3] = Int16.(raw_to_write)
                    end
                    println("  [AUTOSAVE-MASK] Saved mask for TP $tp_i to $ds_path_expert ($(count(>(0), mask_to_save)) non-zero voxels)"); flush(stdout)
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
const _measurement_dirty = Ref(false)

function _ensure_mask_autosave_task!()
    _mask_autosave_task_started[] && return
    _mask_autosave_task_started[] = true
    @async begin
        while true
            sleep(2.0)
            if !isempty(dirty_mask_tps) && !isempty(h5_save_path_ref[])
                flush_all_dirty_masks!()
            end
            # Also autosave measurements if dirty
            if _measurement_dirty[] && !isempty(h5_save_path_ref[])
                try
                    save_measurements_to_h5()
                    _measurement_dirty[] = false
                catch e
                    @warn "Measurement autosave failed" exception=e
                end
            end
        end
    end
end

export mark_measurements_dirty!, save_measurements_to_h5, load_measurements_from_h5!

"""Mark measurements as needing autosave. Called after any measurement change."""
function mark_measurements_dirty!()
    _measurement_dirty[] = true
end

"""Save current measurements to HDF5 as a string attribute on the time point group."""
function save_measurements_to_h5()
    h5_path = h5_save_path_ref[]
    studies_list = studies_ref[]
    if isempty(h5_path) || !isfile(h5_path) || isempty(studies_list)
        println("  [AUTOSAVE-MEAS] Skip: h5_path='$(h5_path)' isfile=$(isfile(h5_path)) nstudies=$(length(studies_list))"); flush(stdout)
        return
    end
    tp_i = current_tp_index[]
    if tp_i < 0 || tp_i >= length(studies_list)
        println("  [AUTOSAVE-MEAS] Skip: tp_i=$tp_i out of range (nstudies=$(length(studies_list)))"); flush(stdout)
        return
    end
    
    # Get measurements from the display objects (stored on panel 1)
    Meas = parentmodule(parentmodule(@__MODULE__)).Measurements
    obj = nothing
    try
        obj = _get_main_display_objects()
    catch; end
    if obj === nothing
        println("  [AUTOSAVE-MEAS] Skip: _main_obj_ref is nothing"); flush(stdout)
        return
    end
    
    serialized = Meas.serialize_measurements(obj.measurements, obj.line_measurements)
    
    study = studies_list[tp_i + 1]
    modality, orig_tp, date_str, ct_fname, pet_fname, mask_fname, node_name, tfm_fname = study[1:8]
    group = tfm_fname == "" ? "BASELINE" : "TFM_" * tfm_fname
    
    lock(_hdf5_io_lock) do
        HDF5.h5open(h5_path, "r+") do h5_file
            if haskey(h5_file, group)
                g = h5_file[group]
                # Write as an attribute on the group
                attr_name = "measurements"
                if haskey(HDF5.attributes(g), attr_name)
                    HDF5.delete_attribute(g, attr_name)
                end
                HDF5.attributes(g)[attr_name] = serialized
                n_spheres = count(m -> !m.is_active, obj.measurements)
                n_lines = count(m -> !m.is_active, obj.line_measurements)
                println("  [AUTOSAVE-MEAS] Saved $(n_spheres) spheres + $(n_lines) lines to $group"); flush(stdout)
            end
        end
    end
end

"""Load measurements from HDF5 for the given time point and populate the display objects."""
function load_measurements_from_h5!(tp_i::Int, obj)
    h5_path = h5_save_path_ref[]
    studies_list = studies_ref[]
    if isempty(h5_path) || !isfile(h5_path) || isempty(studies_list)
        return
    end
    if tp_i < 0 || tp_i >= length(studies_list)
        return
    end
    
    Meas = parentmodule(parentmodule(@__MODULE__)).Measurements
    study = studies_list[tp_i + 1]
    modality, orig_tp, date_str, ct_fname, pet_fname, mask_fname, node_name, tfm_fname = study[1:8]
    group = tfm_fname == "" ? "BASELINE" : "TFM_" * tfm_fname
    
    serialized = ""
    try
        lock(_hdf5_io_lock) do
            HDF5.h5open(h5_path, "r") do h5_file
                if haskey(h5_file, group)
                    g = h5_file[group]
                    if haskey(HDF5.attributes(g), "measurements")
                        serialized = read(HDF5.attributes(g)["measurements"])
                    end
                end
            end
        end
    catch e
        @warn "Failed to load measurements from HDF5" exception=e
    end
    
    if !isempty(serialized)
        spheres, lines = Meas.deserialize_measurements(serialized)
        empty!(obj.measurements)
        append!(obj.measurements, spheres)
        empty!(obj.line_measurements)
        append!(obj.line_measurements, lines)
        println("  [AUTOSAVE-MEAS] Loaded $(length(spheres)) spheres + $(length(lines)) lines from $group"); flush(stdout)
        # Trigger GUI refresh so measurement rows appear
        try
            LMW = _get_lmw()
            if LMW !== nothing
                obs = getfield(LMW, :_lmw_observables)
                if haskey(obs, :obs_refresh_measurements)
                    obs[:obs_refresh_measurements][] = obj
                end
            end
        catch; end
        # Also schedule a delayed refresh in case the GUI wasn't ready
        @async begin
            sleep(2.0)
            try
                LMW2 = _get_lmw()
                if LMW2 !== nothing
                    obs2 = getfield(LMW2, :_lmw_observables)
                    if haskey(obs2, :obs_refresh_measurements)
                        obs2[:obs_refresh_measurements][] = obj
                    end
                end
            catch; end
        end
    end
end

"""Get the panel 1 mainForDisplayObjects. Returns nothing if not available."""
function _get_main_display_objects()
    ch = main_event_channel[]
    # We can't get state objects from here directly — store a ref
    _main_obj_ref[]
end
const _main_obj_ref = Ref{Any}(nothing)
export _main_obj_ref

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
                    already_cached = lock(_tp_cache_lock) do
                        haskey(tp_data_cache, tp)
                    end
                    if !already_cached && tp_loader_ref[] !== nothing
                        t = @elapsed begin
                            entry = lock(_hdf5_io_lock) do
                                # Re-check under IO lock (another load may have raced)
                                cached2 = lock(_tp_cache_lock) do
                                    haskey(tp_data_cache, tp) ? tp_data_cache[tp] : nothing
                                end
                                cached2 !== nothing && return cached2
                                tp_loader_ref[](tp)
                            end
                            if entry !== nothing
                                lock(_tp_cache_lock) do
                                    tp_data_cache[tp] = entry
                                end
                            end
                        end
                        println("  [IO] Preloaded TP $tp in $(round(t, digits=1))s"); flush(stdout)
                    else
                        println("  [IO] TP $tp already cached, skipping"); flush(stdout)
                    end
                elseif msg isa EvictAndPreloadMessage
                    # Evict first to free memory before loading new data
                    lock(_tp_cache_lock) do
                        for tp in msg.evict_tps
                            delete!(tp_data_cache, tp)
                        end
                    end
                    if !isempty(msg.evict_tps)
                        GC.gc(false)
                        println("  [IO] Evicted TPs $(msg.evict_tps)"); flush(stdout)
                    end
                    # Then preload neighbors
                    for tp in msg.preload_tps
                        already_cached = lock(_tp_cache_lock) do
                            haskey(tp_data_cache, tp)
                        end
                        if !already_cached && tp_loader_ref[] !== nothing
                            t = @elapsed begin
                                entry = lock(_hdf5_io_lock) do
                                    cached2 = lock(_tp_cache_lock) do
                                        haskey(tp_data_cache, tp) ? tp_data_cache[tp] : nothing
                                    end
                                    cached2 !== nothing && return cached2
                                    tp_loader_ref[](tp)
                                end
                                if entry !== nothing
                                    lock(_tp_cache_lock) do
                                        tp_data_cache[tp] = entry
                                    end
                                end
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
            if tp_idx == 1 && !lock(_tp_cache_lock) do; haskey(tp_data_cache, tp_idx); end && io_channel[] !== nothing
                try
                    put!(io_channel[], PreloadTPMessage(tp_idx))
                    println("  [STARTUP] Dispatched background preload for TP $tp_idx"); flush(stdout)
                catch; end
            end
        end
    end
end

function get_or_load_tp_data(idx::Int)
    # Fast path: check cache under short lock
    cached = lock(_tp_cache_lock) do
        haskey(tp_data_cache, idx) ? tp_data_cache[idx] : nothing
    end
    cached !== nothing && return cached

    # Slow path: load from HDF5 under IO lock (serializes with save and preload)
    if tp_loader_ref[] !== nothing
        entry = lock(_hdf5_io_lock) do
            # Re-check cache — another thread may have loaded while we waited for the IO lock
            cached2 = lock(_tp_cache_lock) do
                haskey(tp_data_cache, idx) ? tp_data_cache[idx] : nothing
            end
            cached2 !== nothing && return cached2
            tp_loader_ref[](idx)
        end
        if entry !== nothing
            lock(_tp_cache_lock) do
                tp_data_cache[idx] = entry
            end
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

    panel_idx_mapped = panel_idx > 5 ? panel_idx - 5 : panel_idx
    
    panel_voxels = if panel_idx_mapped == 3  # Sagittal (Y,Z,X)
        sz = (size(entry.ct, 2), size(entry.ct, 3), size(entry.ct, 1))
        Any[("CT", PermutedDimsArray(entry.ct, (2,3,1))),
            ("PET", PermutedDimsArray(entry.pet, (2,3,1))),
            ("Mask", PermutedDimsArray(mask_i16, (2,3,1))),
            ("Bone_Overlay", get_or_create_bone_i8(panel_idx, sz)),
            ("Anatomy", anat_i16 !== nothing ? PermutedDimsArray(anat_i16, (2,3,1)) : zeros(Int16, sz))]
    elseif panel_idx_mapped == 4  # Coronal (X,Z,Y)
        sz = (size(entry.ct, 1), size(entry.ct, 3), size(entry.ct, 2))
        Any[("CT", PermutedDimsArray(entry.ct, (1,3,2))),
            ("PET", PermutedDimsArray(entry.pet, (1,3,2))),
            ("Mask", PermutedDimsArray(mask_i16, (1,3,2))),
            ("Bone_Overlay", get_or_create_bone_i8(panel_idx, sz)),
            ("Anatomy", anat_i16 !== nothing ? PermutedDimsArray(anat_i16, (1,3,2)) : zeros(Int16, sz))]
    elseif panel_idx_mapped == 2  # PET-only
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
    
    main_win = if panel_mod in ("T2", "MRI", "MR")
        get(current_windowing, "T2", Float32[0.0, 1000.0])
    elseif panel_mod == "T1"
        get(current_windowing, "T1", Float32[0.0, 600.0])
    elseif panel_mod == "ADC"
        get(current_windowing, "ADC", Float32[0.0, 2200.0])
    elseif panel_mod == "DWI"
        get(current_windowing, "DWI", Float32[0.0, 120.0])
    else
        get(current_windowing, "CT", Float32[-160.0, 240.0])
    end
    
    nuc_win = if panel_mod in ("T2", "MRI", "MR")
        get(current_windowing, "ADC", Float32[0.0, 2200.0])
    elseif panel_mod == "SPECT"
        get(current_windowing, "SPECT", Float32[0.0, 10.0])
    elseif panel_mod in ("ADC", "DWI", "T1")
        main_win
    else
        get(current_windowing, "PET", Float32[0.0, 10.0])
    end
    
    for tex in stateObjects[panel_idx].mainForDisplayObjects.listOfTextSpecifications
        if tex.name == "CT" || (tex.isMainImage && tex.name != "PET" && tex.name != "SPECT")
            tex.minAndMaxValue = Float32.([main_win[1], main_win[2]])
        elseif tex.name == "PET" || tex.name == "SPECT" || tex.isNuclearMask
            tex.minAndMaxValue = Float32.([nuc_win[1], nuc_win[2]])
            # Default PET/nuclear overlay to 0% blend on MRI modalities
            if panel_mod in ("T2", "MRI", "MR", "T1", "ADC", "DWI")
                tex.maskContribution = 0.0f0
            end
        elseif panel_mod in ("T2", "MRI", "MR", "T1", "ADC", "DWI") && (tex.name == "Mask" || tex.name == "segmentation" || (tex.isMultiDiscreteMask && tex.name != "Anatomy" && tex.name != "Bone_Overlay"))
            # Show lesion segments on MRI; show gland (label 4+) only if anatomy toggle is ON
            max_label = 3  # MRI: always clamp to lesion labels 1-3
            T_mm = eltype(tex.minAndMaxValue)
            tex.minAndMaxValue = T_mm.([1, max_label])
            tex.isVisible = true
            if tex.maskContribution <= 0.0f0
                tex.maskContribution = 0.5f0
            end
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
            if isdefined(LMW, :_cached_lesion_ids)
                delete!(LMW._cached_lesion_ids, tp_idx)
            end
            if isdefined(LMW, :_db_dirty)
                LMW._db_dirty[] = true
            end
        end
    catch; end
    delete!(lesion_centroids_cache, (tp_idx, lesion_id))
    delete!(lesion_centroids_cache, lesion_id)
    @debug "  [SUV] Invalidated cache for lesion $lesion_id @ TP $tp_idx"
end

const _async_suv_debounce = Dict{Tuple{Int, Int}, Float64}()

"""
    invalidate_and_recompute_lesion_metrics_async!(lesion_id, tp_idx, mask_vol)

Called when a lesion is painted or modified.
1. Synchronously invalidates caches.
2. Schedules a debounced background task to recompute metrics.
"""
function invalidate_and_recompute_lesion_metrics_async!(lesion_id::Int, tp_idx::Int, mask_vol::Union{AbstractArray, Nothing}=nothing)
    # 1. Invalidate caches (synchronous)
    invalidate_suv_for_lesion(lesion_id, tp_idx)
    
    # Debounce the heavy background task
    now_t = time()
    _async_suv_debounce[(tp_idx, lesion_id)] = now_t
    
    Threads.@spawn begin
        # Wait a short period to batch updates
        sleep(0.5)
        
        # Abort if a newer event arrived
        if get(_async_suv_debounce, (tp_idx, lesion_id), 0.0) > now_t
            return
        end
        
        # 2. Recompute centroid from current mask
        centroid_found = false
        if mask_vol !== nothing
            try
                indices = findall(==(lesion_id), mask_vol)
                if !isempty(indices)
                    cx = round(Int, mean(i[1] for i in indices))
                    cy = round(Int, mean(i[2] for i in indices))
                    cz = round(Int, mean(i[3] for i in indices))
                    lesion_centroids_cache[(tp_idx, lesion_id)] = [cx, cy, cz]
                    if tp_idx == current_tp_index[]
                        lesion_centroids_cache[lesion_id] = [cx, cy, cz]
                    end
                    centroid_found = true
                    @debug "  [SUV] Recomputed centroid for lesion $lesion_id @ TP $tp_idx: ($cx,$cy,$cz)"
                end
            catch e
                @warn "Centroid recompute failed for lesion $lesion_id: $e"
            end
        end
        
        # Try mask_i16 from tp_data_cache
        if !centroid_found
            entry = lock(_tp_cache_lock) do
                haskey(tp_data_cache, tp_idx) ? tp_data_cache[tp_idx] : nothing
            end
            if entry !== nothing
                try
                    m = entry.mask_i16
                    # Optimize: avoid anonymous function overhead for Int16 arrays
                    indices = findall(==(Int16(lesion_id)), m)
                    if !isempty(indices)
                        cx = round(Int, mean(i[1] for i in indices))
                        cy = round(Int, mean(i[2] for i in indices))
                        cz = round(Int, mean(i[3] for i in indices))
                        lesion_centroids_cache[(tp_idx, lesion_id)] = [cx, cy, cz]
                        if tp_idx == current_tp_index[]
                            lesion_centroids_cache[lesion_id] = [cx, cy, cz]
                        end
                        centroid_found = true
                    end
                catch; end
            end
        end
        
        # 3. Update organ mapping from atlas
        try
            atlas = global_ts_atlas[]
            ts_nm = global_ts_names[]
            if atlas !== nothing && ts_nm !== nothing
                organ_name = ""
                # Try volume-based scan first
                if mask_vol !== nothing
                    LA = _get_la()
                    if LA !== nothing
                        organ_name = LA.classify_and_pick_best_organ(mask_vol, atlas, ts_nm, lesion_id)
                    end
                else
                    cached_mask = lock(_tp_cache_lock) do
                        haskey(tp_data_cache, tp_idx) ? tp_data_cache[tp_idx].mask_i16 : nothing
                    end
                    if cached_mask !== nothing
                        LA = _get_la()
                        if LA !== nothing
                            organ_name = LA.classify_and_pick_best_organ(cached_mask, atlas, ts_nm, lesion_id)
                        end
                    end
                end
                
                # Fallback: centroid-based atlas lookup
                if isempty(organ_name) && centroid_found
                    centroid = lesion_centroids_cache[(tp_idx, lesion_id)]
                    mask_sz = lock(_tp_cache_lock) do
                        haskey(tp_data_cache, tp_idx) ? size(tp_data_cache[tp_idx].mask) : nothing
                    end
                    if mask_sz !== nothing
                        sx = clamp(round(Int, centroid[1] * size(atlas,1) / mask_sz[1]), 1, size(atlas,1))
                        sy = clamp(round(Int, centroid[2] * size(atlas,2) / mask_sz[2]), 1, size(atlas,2))
                        sz = clamp(round(Int, centroid[3] * size(atlas,3) / mask_sz[3]), 1, size(atlas,3))
                    else
                        sx = clamp(centroid[1], 1, size(atlas,1))
                        sy = clamp(centroid[2], 1, size(atlas,2))
                        sz = clamp(centroid[3], 1, size(atlas,3))
                    end
                    anat_val = Int(atlas[sx, sy, sz])
                    if anat_val > 0 && haskey(ts_nm, anat_val)
                        organ_name = ts_nm[anat_val]
                    end
                end
                
                if !isempty(organ_name)
                    existing = get(global_organ_mapping[], lesion_id, "")
                    should_update = isempty(existing) || existing in ("Unknown",)
                    if !should_update
                        LA = _get_la()
                        if LA !== nothing
                            new_pri = LA.classify_tissue_priority(organ_name)
                            old_pri = LA.classify_tissue_priority(existing)
                            should_update = new_pri <= old_pri
                        end
                    end
                    if should_update
                        global_organ_mapping[][lesion_id] = organ_name
                        @debug "  [SUV] Auto-mapped lesion $lesion_id → '$organ_name' from paint voxels"
                        try
                            organ_mapping_updated[] = (lesion_id, organ_name)
                            @debug "  [SUV] Fired organ_mapping_updated for lesion $lesion_id → '$organ_name'"
                        catch e
                            @debug "  [SUV] organ_mapping_updated FAILED: $e"
                        end
                    end
                end
            end
        catch e
            @warn "Organ mapping update failed for lesion $lesion_id: $e"
        end
        
        # 4. Recompute SUV/volume
        try
            LMW = _get_lmw()
            if LMW !== nothing
                vol = LMW.compute_lesion_volume(lesion_id, tp_idx)
                suv_str = LMW.compute_lesion_suv_string(lesion_id, tp_idx)
                if !isempty(suv_str)
                    LMW._lesion_suv_cache[(tp_idx, lesion_id)] = suv_str
                end
                @debug "  [SUV] Async recomputed metrics for lesion $lesion_id @ TP $tp_idx: vol=$(round(vol["volume_cc"], digits=2))cc, suv=$(suv_str)"
            end
        catch e
            @warn "Async SUV/volume recompute failed for lesion $lesion_id: $e"
        end
    end
end
const global_bone_atlas = Ref{Any}(nothing)
const global_organ_mapping = Ref{Dict{Int,String}}(Dict{Int,String}())  # lesion_id -> TS organ name (from map_lesions_to_organs)
const organ_mapping_updated = Observable(Tuple{Int,String}((0, "")))  # fires (lesion_id, organ_name) after paint-based mapping
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
# Per-TP segment / lesion names from HDF5 scene hierarchy: tp_index -> Dict{Int, String}(lesion_id -> original_name)
const tp_segment_names = Dict{Int, Dict{Int, String}}()
# Per-TP organ mapping: tp_index -> Dict{Int, String}(lesion_id -> organ_name)
const tp_organ_mapping = Dict{Int, Dict{Int, String}}()

export tp_data_cache, _tp_cache_lock, _hdf5_io_lock, bone_subsegments_cache, lesion_centroids_cache, global_bone_atlas, global_organ_mapping, current_tp_index, tp_labels, tp_descriptions, tp_english_descriptions
export _m2_crosshair_sync, compare_mode, compare_right_tp, tp_switched, get_node_name_for_tp, tp_node_names, _m2_reference_tp
export pet_volumes_cache, global_ts_atlas, global_ts_names, patient_id, h5_path_ref, tp_modalities, volume_z_size, anatomy_labels_cache, tp_segment_names, tp_organ_mapping
export organ_mapping_updated


function reactToChangeTimePoint(data::ChangeTimePointEvent, stateObjects::Vector{StateDataFields})
    t_total = time_ns()
    # Flush any modified masks before changing timepoint or evicting
    if !isempty(dirty_mask_tps); Threads.@spawn flush_all_dirty_masks!(); end
    if isempty(tp_labels)
        @debug "No TP labels loaded. TP navigation disabled."
        return
    end
    
    # Get sorted TP indices
    tp_indices = sort(collect(keys(tp_labels)))
    num_tps = length(tp_indices)
    
    # Save current TP measurements before switching
    try
        if _measurement_dirty[] && _main_obj_ref[] !== nothing
            save_measurements_to_h5()
            _measurement_dirty[] = false
        end
    catch; end
    
    # Find current position in the sorted list
    cur_pos = findfirst(==(current_tp_index[]), tp_indices)
    if cur_pos === nothing
        cur_pos = 1  # default to first TP if current index not found
    end
    
    # Calculate new position with wrapping
    new_pos = mod1(cur_pos + data.change, num_tps)
    new_tp = tp_indices[new_pos]
    current_tp_index[] = new_tp
    
    # Load measurements for the new time point
    try
        if _main_obj_ref[] !== nothing
            load_measurements_from_h5!(new_tp, _main_obj_ref[])
        end
    catch; end
    
    label = get(tp_labels, new_tp, "TP $new_tp")
    @debug "TP Navigation: switching to $label (index=$new_tp)"
    
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
if _m2_reference_tp[] == -1
    next_pos = new_pos < num_tps ? new_pos + 1 : 1
    right_tp = tp_indices[next_pos]
elseif _m2_reference_tp[] == 0
    right_tp = tp_indices[1]
else
    right_tp = _m2_reference_tp[]
end
                # right_tp assigned above
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
            lid_cmp = _clamp_lid_for_tp(current_active_lesion_id[], new_tp)
            reactToSyncLesion(SyncLesionEvent(lid_cmp), stateObjects)
        end
        
        right_label = get(tp_labels, right_tp, "TP $right_tp")
        @debug "Compare: Left=$label, Right=$right_label"
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
                lid = _clamp_lid_for_tp(lid, new_tp)
                try
                    reactToSyncLesion(SyncLesionEvent(lid), stateObjects)
                    @debug "Synced to Lesion $lid for $label"
                catch e
                    @debug "WARNING: Failed to sync Lesion $lid on TP change: $e"
                end
            end
            if DEBUG_VERBOSE[]; println("  [BENCH] bone overlay (reactToSyncLesion): $(round(t_bone_overlay*1000, digits=1))ms"); flush(stdout); end
            # On MRI: force show-all segments (override single-lesion filter from reactToSyncLesion)
            _force_mri_show_all!(stateObjects)
            # Hide/show Panel 2 (PET-only) based on modality
            if !compare_mode[]
                _update_quad_layout_for_modality!(stateObjects, new_tp)
            end
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
            filter!(tp -> !lock(_tp_cache_lock) do; haskey(tp_data_cache, tp); end, neighbors)
            
            # Evict TPs that are far from current (keep current ± 1 only)
            keep_set = Set{Int}([new_tp, tp_indices[prev_pos], tp_indices[next_pos_n]])
            evict_tps = filter(tp -> !in(tp, keep_set), lock(_tp_cache_lock) do; collect(keys(tp_data_cache)); end)
            
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
                cached_entry = lock(_tp_cache_lock) do
                    haskey(tp_data_cache, tp_for_bg) ? tp_data_cache[tp_for_bg] : nothing
                end
                if LMW !== nothing && cached_entry !== nothing
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
                    @debug "  [BG] SUV precomputed for $(length(unique_ids)) lesions"
                end
            catch; end
            
            try
                ct_vol = lock(_tp_cache_lock) do
                    haskey(tp_data_cache, tp_for_bg) ? tp_data_cache[tp_for_bg].ct : nothing
                end
                if ct_vol !== nothing
                    InferenceClient.preload_ct_for_nninteractive(Array{Float32,3}(ct_vol))
                    @debug "[BG] CT preload initiated for $label_for_bg"
                end
            catch; end
        end
    end
    
    t_total_ms = (time_ns() - t_total) / 1e6
    @info "[BENCH] Next/Prev TP Total: $(round(t_total_ms, digits=1))ms"
    tp_switched[] = tp_switched[] + 1
end

function reactToSetTimePoint(data::SetTimePointEvent, stateObjects::Vector{StateDataFields})
    t_total = time_ns()
    if !isempty(dirty_mask_tps); Threads.@spawn flush_all_dirty_masks!(); end
    # Save measurements before TP switch
    try
        if _measurement_dirty[] && _main_obj_ref[] !== nothing
            save_measurements_to_h5()
            _measurement_dirty[] = false
        end
    catch; end
    if isempty(tp_labels)
        @debug "No TP labels loaded. TP navigation disabled."
        return
    end

    tp_indices = sort(collect(keys(tp_labels)))
    num_tps = length(tp_indices)
    target_tp = data.tp_index
    if !haskey(tp_labels, target_tp)
        @debug "Target TP $target_tp not found in loaded labels. Skipping."
        return
    end

    target_pos = findfirst(==(target_tp), tp_indices)
    target_pos = target_pos === nothing ? 1 : target_pos

    if compare_mode[]
        if data.panel == 5
            # Set Right panel (panel 5)
            right_tp = target_tp
            compare_right_tp[] = right_tp
            right_label = get(tp_labels, right_tp, "TP $right_tp")
            @debug "Compare: setting Right to $right_label (index=$right_tp)"
            
            entry_right = get_or_load_tp_data(right_tp)
            if entry_right !== nothing && length(stateObjects) >= 5
                _load_tp_from_entry!(stateObjects, entry_right, 5)
                # Keep panel 5 aligned with panel 1
                stateObjects[5].onScrollData.dimensionToScroll = stateObjects[1].onScrollData.dimensionToScroll
                stateObjects[5].currentDisplayedSlice = stateObjects[1].currentDisplayedSlice
                stateObjects[5].calcDimsStruct.zoom = stateObjects[1].calcDimsStruct.zoom
                stateObjects[5].calcDimsStruct.panX = stateObjects[1].calcDimsStruct.panX
                stateObjects[5].calcDimsStruct.panY = stateObjects[1].calcDimsStruct.panY
                stateObjects[5].isSliceChanged = true
                _force_texture_upload!(stateObjects, 5)
            end
            
            if current_active_lesion_id[] > 0
                lid_cmp = _clamp_lid_for_tp(current_active_lesion_id[], target_tp)
                try reactToSyncLesion(SyncLesionEvent(lid_cmp), stateObjects) catch; end
            end
        else
            # Set Left panel (panel 1)
            left_tp = target_tp
            current_tp_index[] = left_tp
            left_label = get(tp_labels, left_tp, "TP $left_tp")
            @debug "Compare: setting Left to $left_label (index=$left_tp)"
            
            entry_left = get_or_load_tp_data(left_tp)
            if entry_left !== nothing && length(stateObjects) >= 1
                _load_tp_from_entry!(stateObjects, entry_left, 1)
                stateObjects[1].isSliceChanged = true
                _force_texture_upload!(stateObjects, 1)
            end
            
            if current_active_lesion_id[] > 0
                lid_cmp = _clamp_lid_for_tp(current_active_lesion_id[], left_tp)
                try reactToSyncLesion(SyncLesionEvent(lid_cmp), stateObjects) catch; end
            end
        end
    else
        # Single mode: load target_tp into all panels
        current_tp_index[] = target_tp
        # Load measurements for the new time point
        try
            if _main_obj_ref[] !== nothing
                load_measurements_from_h5!(target_tp, _main_obj_ref[])
            end
        catch; end
        label = get(tp_labels, target_tp, "TP $target_tp")
        @debug "TP Navigation: switching to $label (index=$target_tp)"
        
        entry = get_or_load_tp_data(target_tp)
        if entry !== nothing
            for i in 1:min(4, length(stateObjects))
                _load_tp_from_entry!(stateObjects, entry, i)
            end
            if length(stateObjects) >= 5
                _load_tp_from_entry!(stateObjects, entry, 5)
            end
            
            empty!(last_bone_overlay_indices)
            lid = current_active_lesion_id[] > 0 ? current_active_lesion_id[] : 1
            lid = _clamp_lid_for_tp(lid, target_tp)
            try
                reactToSyncLesion(SyncLesionEvent(lid), stateObjects)
                @debug "Synced to Lesion $lid for $label"
            catch e
                @debug "WARNING: Failed to sync Lesion $lid on TP set: $e"
            end
            # On MRI: force show-all segments
            _force_mri_show_all!(stateObjects)
            # Hide/show Panel 2 (PET-only) based on modality
            if !compare_mode[]
                _update_quad_layout_for_modality!(stateObjects, target_tp)
            end
        end
    end

    # Sliding window: preload adjacent TPs (current ± 1), evict distant ones
    if io_channel[] !== nothing
        try
            neighbors = Int[]
            prev_pos = mod1(target_pos - 1, num_tps)
            next_pos_n = mod1(target_pos + 1, num_tps)
            push!(neighbors, tp_indices[prev_pos])
            push!(neighbors, tp_indices[next_pos_n])
            filter!(tp -> !lock(_tp_cache_lock) do; haskey(tp_data_cache, tp); end, neighbors)
            
            keep_set = Set{Int}([target_tp, tp_indices[prev_pos], tp_indices[next_pos_n]])
            if compare_mode[] && compare_right_tp[] >= 0
                push!(keep_set, compare_right_tp[])
            end
            evict_tps = filter(tp -> !in(tp, keep_set), lock(_tp_cache_lock) do; collect(keys(tp_data_cache)); end)
            
            if !isempty(neighbors) || !isempty(evict_tps)
                put!(io_channel[], EvictAndPreloadMessage(evict_tps, neighbors))
            end
        catch; end
    end

    # Background SUV precompute & CT Docker preload
    let tp_for_bg = target_tp, label_for_bg = get(tp_labels, target_tp, "TP $target_tp")
        Threads.@spawn begin
            try
                LMW = _get_lmw()
                cached_entry = lock(_tp_cache_lock) do
                    haskey(tp_data_cache, tp_for_bg) ? tp_data_cache[tp_for_bg] : nothing
                end
                if LMW !== nothing && cached_entry !== nothing
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
                    @debug "  [BG] SUV precomputed for $(length(unique_ids)) lesions"
                end
            catch; end
            
            try
                ct_vol = lock(_tp_cache_lock) do
                    haskey(tp_data_cache, tp_for_bg) ? tp_data_cache[tp_for_bg].ct : nothing
                end
                if ct_vol !== nothing
                    InferenceClient.preload_ct_for_nninteractive(Array{Float32,3}(ct_vol))
                    @debug "[BG] CT preload initiated for $label_for_bg"
                end
            catch; end
        end
    end

    t_total_ms = (time_ns() - t_total) / 1e6
    @info "[BENCH] Set TP Total: $(round(t_total_ms, digits=1))ms"
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
    @debug "Refreshing lesion list..."
end

function reactToAddAutoPet(data::AddAutoPetEvent, stateObjects::Vector{StateDataFields})
    if !InferenceClient.is_ai_enabled()
        set_ai_status!("[AI Disabled] Restart app with AI enabled or run worker on port 5005")
        @debug "[AI] Automatic segmentation requested but AI models are disabled."
        return
    end
    set_ai_status!("[Processing] AI request ($(data.algorithm))...")
    try
        @debug "Add New Lesion (Auto-PET) triggered with algorithm: $(data.algorithm)"
        
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
        
        algo = data.algorithm
        channel = data.channel
        
        # Execute heavy voxel scanning & job queuing asynchronously so consumer / GUI NEVER block
        Threads.@spawn begin
            try
                # Extract scribbles directly from panel 1's mask (seg_vol) —
                # the one 3D array with integer lesion IDs.
                # Only voxels matching active_id are user-painted scribbles.
                if seg_vol === nothing
                    set_ai_status!("[Error] No segmentation mask found for scribble extraction.")
                    return
                end
                
                T_elem = eltype(seg_vol)
                v_act = round(T_elem, active_id)
                v_neg = round(T_elem, -active_id)
                
                painted_pts = findall(seg_vol .== v_act)
                negative_pts = findall(seg_vol .== v_neg)
                
                println("[AI-SCRIBBLE] seg_vol active_id=$active_id: $(length(painted_pts)) pos, $(length(negative_pts)) neg voxels"); flush(stdout)
                
                if isempty(painted_pts)
                    msg = "No painted scribbles found for AI inference. Paint scribbles on the lesion first."
                    @debug "ERROR: $msg"
                    set_ai_status!("[Error] $msg")
                    return
                end
                
                points_vol = zeros(Float32, size(ct_vol))
                for idx in painted_pts
                    if checkbounds(Bool, points_vol, idx)
                        points_vol[idx] = 1.0f0
                    end
                end
                for idx in negative_pts
                    if checkbounds(Bool, points_vol, idx)
                        points_vol[idx] = -1.0f0
                    end
                end
                cx = round(Int, mean([p[1] for p in painted_pts]))
                cy = round(Int, mean([p[2] for p in painted_pts]))
                cz = round(Int, mean([p[3] for p in painted_pts]))
                
                scribble_coords_0idx = [[idx[1]-1, idx[2]-1, idx[3]-1] for idx in painted_pts if checkbounds(Bool, ct_vol, idx)]
                negative_coords_0idx = [[idx[1]-1, idx[2]-1, idx[3]-1] for idx in negative_pts if checkbounds(Bool, ct_vol, idx)]
                
                set_ai_status!("[Preparing] inference ($(algo))...")
                @debug "Queuing $(algo) inference job (seed=$cx,$cy,$cz, lesion=$active_id, $(length(painted_pts)) painted points, $(length(negative_pts)) negative points)..."
                
                # Extract real voxel spacing from scroll dims (critical for nnInteractive autozoom)
                ct_spacing = try
                    tp1_state.onScrollData.dataToScrollDims.voxelSize
                catch
                    (1.0, 1.0, 1.0)
                end
                @debug "[reactToAddAutoPet] Using CT spacing: $ct_spacing"
                
                # Use immutable views / direct references without 680MB deep copies
                put!(inference_queue, InferenceJob(
                    algo, ct_vol, pet_vol, points_vol,
                    cx, cy, cz, active_id, seg_vol, channel, scribble_coords_0idx, negative_coords_0idx, ct_spacing))
            catch e
                err_msg = sprint(showerror, e)
                @debug "ERROR in async reactToAddAutoPet: $err_msg"
                @debug "Error trace" exception=(e, catch_backtrace())
                set_ai_status!("[Error] AI Error: $err_msg")
            end
        end
    catch e
        err_msg = sprint(showerror, e)
        @debug "ERROR in reactToAddAutoPet: $err_msg"
        @debug "Error trace" exception=(e, catch_backtrace())
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
    @debug "AIInferenceResultEvent received: algorithm=$(data.algorithm), active_id=$(data.active_id), seed=($(data.cx),$(data.cy),$(data.cz))"

    if data.mask === nothing
        println("WARNING: AI inference failed or returned nothing."); flush(stdout)
        if !InferenceClient.is_worker_reachable()
            err = InferenceClient.get_last_ai_error()
            msg = isempty(err) ? "[Error] AI worker unreachable on port $(InferenceClient.get_ai_port()). Is Docker running?" : "[Error] AI offline: $err"
            set_ai_status!(msg)
        else
            err = InferenceClient.get_last_ai_error()
            msg = isempty(err) ? "[Warning] Inference failed (no mask returned)" : "[Warning] $err"
            set_ai_status!(msg)
        end
        return
    end

    seg_vol = data.seg_vol
    if seg_vol === nothing
        @debug "ERROR: No segmentation volume reference available. Cannot apply AI results. No fallbacks allowed."
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
    @debug "$(data.algorithm) segmented $(count(data.mask .> 0)) patch voxels for lesion $(data.active_id) at ($(data.cx), $(data.cy), $(data.cz))."
    
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
    entry = lock(_tp_cache_lock) do
        haskey(tp_data_cache, tp_idx) ? tp_data_cache[tp_idx] : nothing
    end
    if entry !== nothing
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
                textSpec.minAndMaxValue = Float32.([0.0, 10000.0])
            end
        end
    end

    # Compute center of the actual segmentation result (not the seed point)
    seg_indices = findall(seg_vol .== eltype(seg_vol)(data.active_id))
    if !isempty(seg_indices)
        center_x = round(Int, mean(i[1] for i in seg_indices))
        center_y = round(Int, mean(i[2] for i in seg_indices))
        center_z = round(Int, mean(i[3] for i in seg_indices))
        @debug "[AI Result] Centering on segmentation centroid: ($center_x, $center_y, $center_z) from $(length(seg_indices)) voxels"
    else
        center_x, center_y, center_z = data.cx, data.cy, data.cz
        @debug "[AI Result] No segmented voxels found, using seed: ($center_x, $center_y, $center_z)"
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
    ReactToScroll.reactToScroll(ScrollEvent(0, 1), stateObjects)
    stateObjects[1].switchIndex = old_sw

    # Update status label
    voxel_count = count(data.mask .> 0)
    set_ai_status!("[Success] Done ($(voxel_count) voxels, lesion $(data.active_id))")

    # Invalidate caches and async-recompute SUV/volume/PROMISE for the modified lesion
    invalidate_and_recompute_lesion_metrics_async!(data.active_id, current_tp_index[], seg_vol)
end
function reactToSyncMissing(data::SyncMissingEvent, stateObjects::Vector{StateDataFields})
    @debug "Sync Missing Lesions across TPs triggered."
    if length(stateObjects) < 2
        @debug "WARNING: Need at least 2 time points to sync missing lesions."
        return
    end
    
    tp1_state = stateObjects[1]
    tp2_state = stateObjects[2]
    
    # For now, just sync the current position. A full sync would iterate over all unique values in tp1_seg.
    pos = tp1_state.lastRecordedMousePosition
    
    tp2_ct = tp2_state.mainForDisplayObjects.listOfTextSpecifications[1].imageTexture
    tp2_pet = tp2_state.mainForDisplayObjects.listOfTextSpecifications[2].imageTexture
    tp2_seg = tp2_state.mainForDisplayObjects.listOfTextSpecifications[3].imageTexture
    
    @debug "Running HelpNet on TP2 for missing lesion at $pos..."
    mask = InferenceClient.run_helpnet_inference(tp2_ct, tp2_pet, pos[1], pos[2], pos[3])
    if mask !== nothing
        InferenceClient.insert_patch!(tp2_seg, mask, pos[1], pos[2], pos[3])
        LesionAssociation.map_link("TP1", "TP2", "SyncedLesion")
        @debug "Successfully synced and mapped lesion to TP2."
    end
end

function reactToGenManual(data::GenManualEvent, stateObjects::Vector{StateDataFields})
    @debug "Bone subsegmentation (manual) triggered for lesion $(data.lesion_id)"
    
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
    @debug "Map Link triggered. Linking lesions: src=$(data.src_ids) to dst=$(data.dst_ids)"
    if length(stateObjects) > 1
        # LesionAssociation.map_link("TP1", "TP2", data.src_ids, data.dst_ids)
        @debug "Successfully mapped between TP1 and TP2"
    end
end

function reactToAutoRunPreprocess(data::AutoRunPreprocessEvent, stateObjects::Vector{StateDataFields})
    @debug "Auto-run preprocessing toggled to $(data.active)"
end

function reactToRunPreprocess(data::RunPreprocessEvent, stateObjects::Vector{StateDataFields})
    @debug "Full Preprocessing triggered."
end

function reactToShowBoneMask(data::ShowBoneMaskEvent, stateObjects::Vector{StateDataFields})
    @debug "Show Bone Mask toggled to $(data.active)"
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
            ReactToScroll.reactToScroll(ScrollEvent(0, p > 5 ? 2 : 1), stateObjects, false)
        end
    end
    stateObjects[1].switchIndex = old_sw
end

const MASK_BACKUP = Dict{UInt64, Array{Float32, 3}}()

function reactToShowMaskLayer(data::ShowMaskLayerEvent, stateObjects::Vector{StateDataFields})
    @debug "reactToShowMaskLayer: layer=$(data.layer) active=$(data.active)"
    
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
                @debug "  Panel $si: set isVisible=$(data.active) for texture $(textSpec.name)"
            end
        end
    end
    @debug "  Toggled $toggled_count textures for layer=$(data.layer)"
    
    # If bone surface or marrow is toggled, also sync dataToScroll buffer directly
    if data.layer == 2 || data.layer == 3
        cur_lid = (current_active_lesion_id[] > 0) ? current_active_lesion_id[] : round(Int, stateObjects[1].valueForMasToSet.value)
        @debug "  Bone data sync: cur_lid=$cur_lid"
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
    
    # On MRI modalities: update Mask label range to show/hide prostate gland (label 4+)
    if data.layer == 4
        tp = current_tp_index[]
        panel_mod = uppercase(get(tp_modalities, tp, "PET"))
        if panel_mod in ("T2", "MRI", "MR", "T1", "ADC", "DWI")
            _force_mri_show_all!(stateObjects)
            # Force re-render to reflect the updated mask range
            for idx in visible_panels
                stateObjects[idx].isSliceChanged = true
                if stateObjects[idx].mainForDisplayObjects.vulkanPipelineState !== nothing
                    stateObjects[idx].mainForDisplayObjects.vulkanPipelineState.ubo_dirty = true
                end
            end
        end
    end
end

function reactToSaveMRB(data::SaveMRBEvent, stateObjects::Vector{StateDataFields})
    @debug "Save MRB triggered: saving all dirty masks to HDF5..."
    flush_all_dirty_masks!()
end

export reactToToggleMoveLesionMode
function reactToToggleMoveLesionMode(data::ToggleMoveLesionModeEvent, stateObjects::Vector{StateDataFields})
    @debug "Move Lesion Mode toggled to $(data.active)"
    for state in stateObjects
        state.moveLesionMode = data.active
    end
end

atexit() do
    flush_all_dirty_masks!()
    try
        if _measurement_dirty[] && _main_obj_ref[] !== nothing
            save_measurements_to_h5()
        end
    catch; end
end

function reactToSetM2Reference(data::SetM2ReferenceEvent, stateObjects::Vector{StateDataFields})
    _m2_reference_tp[] = data.tp_index
    @info "Set M2 reference TP to $(data.tp_index)"
    
    if compare_mode[] || _flicker_active[]
        # Update current compare_right_tp to reflect new reference choice
        tp_indices = sort(collect(keys(tp_labels)))
        if !isempty(tp_indices)
            cur_pos = findfirst(==(current_tp_index[]), tp_indices)
            cur_pos = cur_pos === nothing ? 1 : cur_pos
            if _m2_reference_tp[] == -1
                prev_pos = cur_pos > 1 ? cur_pos - 1 : length(tp_indices)
                right_tp = tp_indices[prev_pos]
            elseif _m2_reference_tp[] == 0
                right_tp = tp_indices[1]
            else
                right_tp = _m2_reference_tp[]
            end
            
            if right_tp != compare_right_tp[]
                compare_right_tp[] = right_tp
                entry_right = get_or_load_tp_data(right_tp)
                if entry_right !== nothing && length(stateObjects) >= 5
                    _load_tp_from_entry!(stateObjects, entry_right, 5)
                end
            end
            # Sync GUI right TP dropdown
            try
                LMW = _get_lmw()
                if LMW !== nothing
                    obs_dict = getfield(LMW, :_lmw_observables)
                    if haskey(obs_dict, :menu_tp_right)
                        menu = obs_dict[:menu_tp_right]
                        tp_label = get(tp_labels, right_tp, "TP $(right_tp)")
                        idx = findfirst(==(tp_label), menu.options[])
                        if idx !== nothing
                            menu.i_selected[] = idx
                            menu.selection[] = menu.options[][idx]
                        end
                    end
                    # Force Makie redraw (safe — no visibility toggling)
                    _safe_redraw(obs_dict)
                end
            catch; end
        end
    end
end
export reactToSetM2Reference

# ─── Measurement Mode Handlers ──────────────────────────────────────────

export reactToToggleMeasurementMode, reactToJumpToMeasurement, reactToDeleteMeasurement

function reactToToggleMeasurementMode(data::MakieEvents.ToggleMeasurementModeEvent, stateObjects::Vector{StateDataFields})
    measurements_mode[] = !measurements_mode[]
    if !measurements_mode[]
        editing_measurement_id[] = 0
        editing_measurement_type[] = :none
    end
    @info "Measurement mode: $(measurements_mode[] ? "ON" : "OFF")"
    
    # Notify LMW GUI if available
    try
        LMW = _get_lmw()
        if LMW !== nothing
            obs_dict = getfield(LMW, :_lmw_observables)
            if haskey(obs_dict, :obs_measurement_mode_changed)
                obs_dict[:obs_measurement_mode_changed][] = obs_dict[:obs_measurement_mode_changed][] + 1
            end
        end
    catch; end
end

function reactToJumpToMeasurement(data::MakieEvents.JumpToMeasurementEvent, stateObjects::Vector{StateDataFields})
    if isempty(stateObjects) || length(stateObjects) < 4
        return
    end
    
    Meas = parentmodule(parentmodule(@__MODULE__)).Measurements
    obj = stateObjects[1].mainForDisplayObjects
    
    idx = findfirst(m -> m.id == data.id, obj.measurements)
    if idx !== nothing
        m = obj.measurements[idx]
        cx, cy, cz = round(Int, m.center_idx[1]), round(Int, m.center_idx[2]), round(Int, m.center_idx[3])
    else
        idx_line = findfirst(m -> m.id == data.id, obj.line_measurements)
        if idx_line !== nothing
            m = obj.line_measurements[idx_line]
            cx, cy, cz = round(Int, (m.start_idx[1] + m.end_idx[1])/2), round(Int, (m.start_idx[2] + m.end_idx[2])/2), round(Int, (m.start_idx[3] + m.end_idx[3])/2)
        else
            @warn "Measurement #$(data.id) not found"
            return
        end
    end
    
    @info "Jumping to measurement #$(data.id) at voxel ($cx, $cy, $cz)"
    
    # Jump all panels to the measurement center
    # Panel 1/2/5: Axial, scroll Z
    # Panel 3: Sagittal, scroll X
    # Panel 4: Coronal, scroll Y
    targets = Dict{Int,Int}(1 => cz, 2 => cz, 3 => cx, 4 => cy)
    if length(stateObjects) >= 5
        targets[5] = cz
    end
    
    ReactToScroll = parentmodule(parentmodule(@__MODULE__)).ReactToScroll
    ReactToScroll.reactToScrollMultiPanel!(collect(keys(targets)), stateObjects, targets)
    current_viewer_position[] = (cx, cy, cz)

end

function reactToDeleteMeasurement(data::MakieEvents.DeleteMeasurementEvent, stateObjects::Vector{StateDataFields})
    if isempty(stateObjects)
        return
    end
    
    obj = stateObjects[1].mainForDisplayObjects
    if data.id == -1
        # Clear all measurements
        for so in stateObjects
            empty!(so.mainForDisplayObjects.measurements)
        end
        @info "Cleared all measurements"
    else
        idx = findfirst(m -> m.id == data.id, obj.measurements)
        if idx !== nothing
            deleteat!(obj.measurements, idx)
            @info "Deleted measurement #$(data.id)"
        end
    end
    if data.id == -1 || editing_measurement_id[] == data.id
        editing_measurement_id[] = 0
        editing_measurement_type[] = :none
    end
    mark_measurements_dirty!()
    
    try
        LMW = _get_lmw()
        if LMW !== nothing
            obs = getfield(LMW, :_lmw_observables)
            if haskey(obs, :obs_refresh_measurements)
                obs[:obs_refresh_measurements][] = obj
            end
        end
    catch; end
end

# ─── Line Measurement Handlers ──────────────────────────────────────────

export reactToCycleMeasurementSubMode, reactToDeleteLineMeasurement, reactToJumpToLineMeasurement

function reactToCycleMeasurementSubMode(data::MakieEvents.CycleMeasurementSubModeEvent, stateObjects::Vector{StateDataFields})
    measurement_sub_mode[] = measurement_sub_mode[] == :sphere ? :line : :sphere
    @info "Measurement sub-mode: $(measurement_sub_mode[])"
    
    # Notify GUI
    try
        LMW = _get_lmw()
        if LMW !== nothing
            obs_dict = getfield(LMW, :_lmw_observables)
            if haskey(obs_dict, :obs_measurement_mode_changed)
                obs_dict[:obs_measurement_mode_changed][] = obs_dict[:obs_measurement_mode_changed][] + 1
            end
        end
    catch; end
end

function reactToDeleteLineMeasurement(data::MakieEvents.DeleteLineMeasurementEvent, stateObjects::Vector{StateDataFields})
    if isempty(stateObjects)
        return
    end
    obj = stateObjects[1].mainForDisplayObjects
    if data.id == -1
        for so in stateObjects
            empty!(so.mainForDisplayObjects.line_measurements)
        end
        @info "Cleared all line measurements"
    else
        idx = findfirst(m -> m.id == data.id, obj.line_measurements)
        if idx !== nothing
            deleteat!(obj.line_measurements, idx)
            @info "Deleted line measurement #$(data.id)"
        end
    end
    if data.id == -1 || editing_measurement_id[] == data.id
        editing_measurement_id[] = 0
        editing_measurement_type[] = :none
    end
    mark_measurements_dirty!()
    
    try
        LMW = _get_lmw()
        if LMW !== nothing
            obs = getfield(LMW, :_lmw_observables)
            if haskey(obs, :obs_refresh_measurements)
                obs[:obs_refresh_measurements][] = obj
            end
        end
    catch; end
end

function reactToJumpToLineMeasurement(data::MakieEvents.JumpToLineMeasurementEvent, stateObjects::Vector{StateDataFields})
    if isempty(stateObjects) || length(stateObjects) < 4
        return
    end
    
    obj = stateObjects[1].mainForDisplayObjects
    idx = findfirst(m -> m.id == data.id, obj.line_measurements)
    if idx === nothing
        @warn "Line measurement #$(data.id) not found"
        return
    end
    
    lm = obj.line_measurements[idx]
    # Jump to midpoint of the line
    mx = round(Int, (lm.start_idx[1] + lm.end_idx[1]) / 2)
    my = round(Int, (lm.start_idx[2] + lm.end_idx[2]) / 2)
    mz = round(Int, (lm.start_idx[3] + lm.end_idx[3]) / 2)
    
    @info "Jumping to line measurement #$(data.id) midpoint ($mx, $my, $mz)"
    
    targets = Dict{Int,Int}(1 => mz, 2 => mz, 3 => mx, 4 => my)
    if length(stateObjects) >= 5
        targets[5] = mz
    end
    
    ReactToScroll = parentmodule(parentmodule(@__MODULE__)).ReactToScroll
    ReactToScroll.reactToScrollMultiPanel!(collect(keys(targets)), stateObjects, targets)
    current_viewer_position[] = (mx, my, mz)

end

# ─── Edit Measurement Handlers ──────────────────────────────────────────────
# Observable to track which measurement is in "edit" mode (ready for edge/endpoint drag)
const editing_measurement_id = Observable{Int}(0)   # 0 = none
const editing_measurement_type = Observable{Symbol}(:none)  # :sphere or :line
export editing_measurement_id, editing_measurement_type
export reactToEditMeasurement, reactToEditLineMeasurement

function reactToEditMeasurement(data::MakieEvents.EditMeasurementEvent, stateObjects::Vector{StateDataFields})
    if isempty(stateObjects); return; end
    obj = stateObjects[1].mainForDisplayObjects
    idx = findfirst(m -> m.id == data.id, obj.measurements)
    if idx === nothing; return; end
    
    m = obj.measurements[idx]
    # Enable measurement mode + sphere sub-mode
    measurements_mode[] = true
    measurement_sub_mode[] = :sphere
    # Initialize per-axis radii from base radius
    if m.radius_x_mm <= 0; m.radius_x_mm = m.radius_mm; end
    if m.radius_y_mm <= 0; m.radius_y_mm = m.radius_mm; end
    if m.radius_z_mm <= 0; m.radius_z_mm = m.radius_mm; end
    # Mark as editing — next click near edge will start drag
    editing_measurement_id[] = data.id
    editing_measurement_type[] = :sphere
    println("  [MEAS-EDIT] Sphere #$(data.id) ready for editing"); flush(stdout)
    
    # Jump to the sphere center
    cx, cy, cz = round(Int, m.center_idx[1]), round(Int, m.center_idx[2]), round(Int, m.center_idx[3])
    targets = Dict{Int,Int}()
    for i in 1:min(2, length(stateObjects)); targets[i] = cz; end
    if length(stateObjects) >= 3; targets[3] = cx; end
    if length(stateObjects) >= 4; targets[4] = cy; end
    if length(stateObjects) >= 5; targets[5] = cz; end
    ReactToScroll = parentmodule(parentmodule(@__MODULE__)).ReactToScroll
    ReactToScroll.reactToScrollMultiPanel!(collect(keys(targets)), stateObjects, targets)
end

function reactToEditLineMeasurement(data::MakieEvents.EditLineMeasurementEvent, stateObjects::Vector{StateDataFields})
    if isempty(stateObjects); return; end
    obj = stateObjects[1].mainForDisplayObjects
    idx = findfirst(m -> m.id == data.id, obj.line_measurements)
    if idx === nothing; return; end
    
    lm = obj.line_measurements[idx]
    # Enable measurement mode + line sub-mode
    measurements_mode[] = true
    measurement_sub_mode[] = :line
    # Mark as editing
    editing_measurement_id[] = data.id
    editing_measurement_type[] = :line
    println("  [MEAS-EDIT] Line #$(data.id) ready for editing"); flush(stdout)
    
    # Jump to the line midpoint
    mx = round(Int, (lm.start_idx[1] + lm.end_idx[1]) / 2)
    my = round(Int, (lm.start_idx[2] + lm.end_idx[2]) / 2)
    mz = round(Int, (lm.start_idx[3] + lm.end_idx[3]) / 2)
    targets = Dict{Int,Int}()
    for i in 1:min(2, length(stateObjects)); targets[i] = mz; end
    if length(stateObjects) >= 3; targets[3] = mx; end
    if length(stateObjects) >= 4; targets[4] = my; end
    if length(stateObjects) >= 5; targets[5] = mz; end
    ReactToScroll = parentmodule(parentmodule(@__MODULE__)).ReactToScroll
    ReactToScroll.reactToScrollMultiPanel!(collect(keys(targets)), stateObjects, targets)
end

end
