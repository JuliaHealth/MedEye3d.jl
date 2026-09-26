

"""
module
code adapted from https://discourse.julialang.org/t/custom-subject-in-rocket-jl-for-mouse-events-from-glfw/65133/3
it is design to help processing data from
    -GLFW.SetCursorPosCallback(window, (_, x, y) -> println("cursor: x, y")) and  for example : cursor: 29.0, 469.0  types   Float64  Float64
    -GLFW.SetMouseButtonCallback(window, (_, button, action, mods) -> println("button action"))  for example types MOUSE_BUTTON_1 PRESS   GLFW.MouseButton  GLFW.Action
The main function is to mark the interaction of the mouse to be saved in appropriate mask and be rendered onto the screen
so we modify the data that is the basis of the mouse interaction mask  and we pass the data on so appropriate part of the texture would be modified to be displayed on screen

"""
module ReactOnMouseClickAndDrag
using Logging, Parameters, Setfield, GLFW, Dates, Parameters, Logging, Base.Threads
using ..ForDisplayStructs, ..TextureManag, ..OpenGLDisplayUtils
using ..DataStructs, ..StructsManag, ..ShadersAndVerticiesForLine, ..ReactToScroll, ..DisplayWords, ..StrokeRasterization
import Logging, Base.Threads
export registerMouseClickFunctions
export reactToMouseDrag
export react_to_draw
export reactToDoubleClick
export DoubleClickEvent

"""
Calculates OpenGl coordinate system values for
    left edge of the text area,
    middle point of the image region,
    range of values for the image
"""
#careful fraction of mainImage should not be grather than 1
function openGlSystemVals(fractionOfMainImage::Float32, windowWidth::Int)
    #Working in normal coordinate system 0 -1 for calculating mid point

    windowWidthLowerBound = 0
    windowWidthUpperBound = windowWidth
    openGlLowerBound = -1
    openGlUpperBound = 1

    textAreaBegin = fractionOfMainImage * windowWidth
    imageRange = windowWidthUpperBound - windowWidthLowerBound
    imageMidPoint = (imageRange * fractionOfMainImage) / 2

    #if ratio of mainIMage is 0.8 , gives us (0.8/1 *2 ) -1 = 0.6
    textBeginningOpenGl = ((textAreaBegin / imageRange) * 2) - 1
    #gives us -0.2
    imageMidPointOpenGl = (imageMidPoint / imageRange) * 2 - 1
    openGlImageRange = textBeginningOpenGl - openGlLowerBound

    return (textBeginningOpenGl, imageMidPointOpenGl, openGlImageRange)
end

"""
we pass coordinate of cursor only when isLeftButtonDown is true and we make it true
if left button is presed down - we make it true if the left button is pressed over image and false if mouse get out of the window or we get information about button release
imageWidth adn imageHeight are the dimensions of textures that we use to display
"""
# Module-level timestamp for double-click detection (avoids GLFW.GetTime which doesn't exist in Julia GLFW.jl)
const lastLeftClickTimestamp = Ref{Float64}(0.0)

function registerMouseClickFunctions(window::GLFW.Window, calcD::CalcDimsStruct, mainChannel::Base.Channel{Any}, window_id::Int=1)
    xmin = Int32(calcD.windowWidthCorr)
    xmax = Int32(calcD.avWindWidtForMain - calcD.windowWidthCorr)

    ymin = Int32(calcD.windowHeightCorr)
    ymax = Int32(calcD.avWindHeightForMain - calcD.windowHeightCorr)
    # calculating dimensions of quad becouse it do not occupy whole window, and we want to react only to those mouse positions that are on main image quad
    mouseStructInstance = MouseStruct()
    mouseStructInstance.window_id = window_id
    
    # Query actual GLFW window size (may differ from requested size due to WM resize)
    actualW, actualH = GLFW.GetWindowSize(window)
    mouseStructInstance.actualWindowWidth = Int(actualW)
    mouseStructInstance.actualWindowHeight = Int(actualH)
    @info "GLFW actual window size: $(actualW)x$(actualH) vs stored: $(calcD.windowWidth)x$(calcD.windowHeight)"

    GLFW.SetCursorPosCallback(window, (a, x, y) -> begin
        # Use cached window dimensions (updated by FramebufferSizeCallback / button callback)
        # instead of querying GLFW.GetWindowSize on every pixel of mouse movement
        aW = mouseStructInstance.actualWindowWidth
        aH = mouseStructInstance.actualWindowHeight
        
        if (x >= 0 && x <= aW && y >= 0 && y <= aH)
            point = CartesianIndex(Int(x), Int(y))
            mouseStructInstance.lastCoordinates = [point]
            # Snapshot into a new struct so later callbacks cannot overwrite this message
            
            # Anti-deadlock: drop rapid mouse movements if consumer channel is getting full.
            # If we block here, we deadlock the Makie renderloop (which holds GLOBAL_OPENGL_LOCK)
            # against the consumer thread which might be loading a large HDF5 file.
            if isready(mainChannel) && length(mainChannel.data) >= 950
                return # drop mouse move event
            end
            
            put!(mainChannel, MouseStruct(
                isLeftButtonDown  = mouseStructInstance.isLeftButtonDown,
                isRightButtonDown = mouseStructInstance.isRightButtonDown,
                lastCoordinates   = [point],
                actualWindowWidth  = aW,
                actualWindowHeight = aH,
                window_id = window_id
            ))
        end
    end)# and  for example : cursor: 29.0, 469.0  types   Float64  Float64
    GLFW.SetMouseButtonCallback(window, (a, button, action, mods) -> begin
        # Refresh cached window dimensions (infrequent: only on button press/release)
        try
            bW, bH = GLFW.GetWindowSize(window)
            mouseStructInstance.actualWindowWidth = Int(bW)
            mouseStructInstance.actualWindowHeight = Int(bH)
        catch end
        
        # Only update the flag for the button that actually changed
        if button == GLFW.MOUSE_BUTTON_1
            mouseStructInstance.isLeftButtonDown = (action == GLFW.PRESS)
        elseif button == GLFW.MOUSE_BUTTON_2
            mouseStructInstance.isRightButtonDown = (action == GLFW.PRESS)
        end
        
        leftMouseButtonDownResult = (button == GLFW.MOUSE_BUTTON_1 && action == GLFW.PRESS)

        # Double-click detection: fire a dedicated DoubleClickEvent into the channel.
        # Uses time() (Base Julia) — GLFW.GetTime() does not exist in Julia's GLFW.jl.
        if leftMouseButtonDownResult
            now = time()
            if (now - lastLeftClickTimestamp[]) < 0.35  # 350ms threshold
                # Fire DoubleClickEvent — its own dispatch type, not embedded in MouseStruct
                coords = mouseStructInstance.lastCoordinates
                put!(mainChannel, DoubleClickEvent(
                    x = isempty(coords) ? 0 : coords[1][1],
                    y = isempty(coords) ? 0 : coords[1][2],
                    actualWindowWidth  = mouseStructInstance.actualWindowWidth,
                    actualWindowHeight = mouseStructInstance.actualWindowHeight,
                    window_id = window_id
                ))
            end
            lastLeftClickTimestamp[] = now
        end

        # Snapshot regular mouse event (for right-click and position tracking)
        put!(mainChannel, MouseStruct(
            isLeftButtonDown  = mouseStructInstance.isLeftButtonDown,
            isRightButtonDown = mouseStructInstance.isRightButtonDown,
            lastCoordinates   = mouseStructInstance.lastCoordinates,
            actualWindowWidth  = mouseStructInstance.actualWindowWidth,
            actualWindowHeight = mouseStructInstance.actualWindowHeight,
            window_id = window_id
        ))
    end) # for example types MOUSE_BUTTON_1 PRESS   GLFW.MouseButton  GLFW.Action

end #registerMouseScrollFunctions



mouseCoords_channel = Base.Channel{MouseStruct}(100)
# we can fetch! on the channel, what is the next thing line, if the mouseStruct, check previous one by fetch. If it mouseStruct, aggregate those 2 and fetch the next one
#fetch in while loop, until no more mouseStructs, then we have the last one, and we can react to it



"""
Determine which panel was clicked/hovered based on mouse cursor position and window layout.
"""
function _detect_clicked_panel(x::Real, y::Real, actualW::Real, actualH::Real, window_id::Int, mainStates::Vector{StateDataFields})::Int
    if window_id == 2 && length(mainStates) >= 10
        is_m2_two_panel = (length(mainStates) >= 8 &&
            (!isempty(mainStates[8].calcDimsStruct.mainImageQuadVert) &&
             mainStates[8].calcDimsStruct.mainImageQuadVert[1] == 0.0f0 &&
             mainStates[8].calcDimsStruct.mainImageQuadVert[2] == 0.0f0))
        if is_m2_two_panel
            return x < actualW / 2.0 ? 6 : 7
        else
            if x < actualW / 2.0 && y < actualH / 2.0
                return 6
            elseif x >= actualW / 2.0 && y < actualH / 2.0
                return 7
            elseif x < actualW / 2.0 && y >= actualH / 2.0
                return 8
            else
                return 9
            end
        end
    else
        is_compare = (length(mainStates) >= 5 &&
            (!isempty(mainStates[3].calcDimsStruct.mainImageQuadVert) &&
             mainStates[3].calcDimsStruct.mainImageQuadVert[1] == 0.0f0 &&
             mainStates[3].calcDimsStruct.mainImageQuadVert[2] == 0.0f0))
        if is_compare
            return x < actualW / 2.0 ? 1 : 5
        elseif length(mainStates) >= 4
            if x < actualW / 2.0 && y < actualH / 2.0
                return 1
            elseif x >= actualW / 2.0 && y < actualH / 2.0
                return 2
            elseif x < actualW / 2.0 && y >= actualH / 2.0
                return 3
            else
                return 4
            end
        elseif length(mainStates) > 1
            return x < actualW / 2.0 ? 1 : 2
        else
            return 1
        end
    end
end

"""
used when we want to save some manual modifications
"""
function react_to_draw(mouseStructArray::Vector{MouseStruct}, mainStates::Vector{StateDataFields})
    if isempty(mouseStructArray)
        return
    end

    # First, detect the active panel from the first sampled point
    first_mouse = mouseStructArray[1]
    if !isempty(first_mouse.lastCoordinates)
        x, y = first_mouse.lastCoordinates[1][1], first_mouse.lastCoordinates[1][2]
        win_idx = clamp(first_mouse.window_id, 1, 2)
        zoomState = quadZoomStates[win_idx]
        if zoomState.isZoomed
            mainStates[1].switchIndex = zoomState.zoomedPanel
        else
            viewportW = Float64(mainStates[1].calcDimsStruct.windowWidth)
            viewportH = Float64(mainStates[1].calcDimsStruct.windowHeight)
            actualW = first_mouse.actualWindowWidth > 0 ? Float64(first_mouse.actualWindowWidth) : viewportW
            actualH = first_mouse.actualWindowHeight > 0 ? Float64(first_mouse.actualWindowHeight) : viewportH
            
            mainStates[1].switchIndex = _detect_clicked_panel(x, y, actualW, actualH, first_mouse.window_id, mainStates)
        end
    end

    stateObject = mainStates[mainStates[1].switchIndex]
    
    # --- MEASUREMENT INTERCEPT (DRAG) ---
    MEH = parentmodule(@__MODULE__).SegmentationDisplay.MakieEventHandlers
    sub_mode = MEH.measurement_sub_mode[]
    meas_on = MEH.measurements_mode[]
    if meas_on && (sub_mode == :sphere || sub_mode == :line)
        MeasMod = parentmodule(@__MODULE__).Measurements
        calcDim = stateObject.calcDimsStruct
        first_mouse = mouseStructArray[end]
        if isempty(first_mouse.lastCoordinates); return; end
        actualW = first_mouse.actualWindowWidth > 0 ? Float64(first_mouse.actualWindowWidth) : Float64(calcDim.windowWidth)
        actualH = first_mouse.actualWindowHeight > 0 ? Float64(first_mouse.actualWindowHeight) : Float64(calcDim.windowHeight)
        
        c = first_mouse.lastCoordinates[1]
        texX, texY = StructsManag.getTextureCoordinatesFromScreen(c[1], c[2], calcDim, actualW, actualH)
        
        ix, iy = round(Int, texX), round(Int, texY)
        clickedPanel = mainStates[1].switchIndex
        slice = stateObject.currentDisplayedSlice
        
        cp_mapped = clickedPanel > 5 ? clickedPanel - 5 : clickedPanel
        if cp_mapped == 3 # Sagittal
            origX, origY, origZ = slice, ix, iy
        elseif cp_mapped == 4 # Coronal
            origX, origY, origZ = ix, slice, iy
        else # Axial (1,2,5)
            origX, origY, origZ = ix, iy, slice
        end
        
        # Clean up active ghost measurements from other panels
        for i in 1:length(mainStates)
            if i != clickedPanel
                filter!(m -> !m.is_active, mainStates[i].mainForDisplayObjects.measurements)
                filter!(m -> !m.is_active, mainStates[i].mainForDisplayObjects.line_measurements)
            end
        end
        
        # Always store measurements on panel 1's display object for cross-view visibility
        meas_obj = mainStates[1].mainForDisplayObjects
        if sub_mode == :sphere
            _handle_sphere_measurement(first_mouse, mainStates, meas_obj, (Float32(origX), Float32(origY), Float32(origZ)), MEH, MeasMod)
        elseif sub_mode == :line
            _handle_line_measurement(first_mouse, mainStates, meas_obj, (Float32(origX), Float32(origY), Float32(origZ)), MEH, MeasMod)
        end
        return # Do not paint
    end
    # --- END MEASUREMENT INTERCEPT ---

    if !stateObject.valueForMasToSet.is_painting_active || isempty(stateObject.textureToModifyVec)
        return
    end
    texture = stateObject.textureToModifyVec[1]
    calcDim = stateObject.calcDimsStruct

    # Extract all sampled mouse coordinates in texture pixel space
    sampledPoints = Tuple{Int,Int}[]
    for mouseStruct in mouseStructArray
        mouseCoords = mouseStruct.lastCoordinates
        actualW = Float64(mouseStruct.actualWindowWidth)
        actualH = Float64(mouseStruct.actualWindowHeight)
        for c in mouseCoords
            texX, texY = StructsManag.getTextureCoordinatesFromScreen(c[1], c[2], calcDim, actualW, actualH)
            ix, iy = Int(round(texX)), Int(round(texY))
            if ix >= 1 && ix <= calcDim.imageTextureWidth && iy >= 1 && iy <= calcDim.imageTextureHeight
                push!(sampledPoints, (ix, iy))
            end
        end
    end

    if isempty(sampledPoints)
        return
    end

    # Build polyline: connect from last point of previous frame if on the same slice
    pointsToRasterize = Tuple{Int,Int}[]
    if !stateObject.isSliceChanged && !isempty(stateObject.lastPaintCoords)
        push!(pointsToRasterize, (stateObject.lastPaintCoords[1][1], stateObject.lastPaintCoords[1][2]))
    end
    append!(pointsToRasterize, sampledPoints)
    stateObject.isSliceChanged = false

    # Store last point for next frame
    stateObject.lastPaintCoords = [CartesianIndex(sampledPoints[end][1], sampledPoints[end][2])]

    # Access current slice data
    if !haskey(stateObject.currentlyDispDat.nameIndexes, texture.name)
        println("[PAINT-DBG] ERROR: texture '$(texture.name)' not found in nameIndexes: $(collect(keys(stateObject.currentlyDispDat.nameIndexes)))"); flush(stdout)
        return
    end
    twoDimDat = stateObject.currentlyDispDat |>
                (singSl) -> singSl.listOfDataAndImageNames[singSl.nameIndexes[texture.name]]

    is_ctrl = mainStates[1].fieldKeyboardStruct.isCtrlPressed
    val = stateObject.valueForMasToSet.value
    if is_ctrl && val > 0
        val = -val
    end
    toSet = convert(twoDimDat.type, convert(parameter_type(texture), val))
    strokeW = Int(texture.strokeWidth)

    # ── AI Mask Immutability: Backup original AI mask before first expert edit ──
    MEH = parentmodule(@__MODULE__).SegmentationDisplay.MakieEventHandlers
    tp_idx = MEH.current_tp_index[]
    raw_val = round(Int, stateObject.valueForMasToSet.value)
    lesion_id = raw_val > 0 ? raw_val : MEH.current_active_lesion_id[]
    if lesion_id <= 0
        lesion_id = MEH.current_active_lesion_id[]
    end
    if lesion_id > 0
        try
            MEH.ensure_ai_mask_backup!(lesion_id, tp_idx, stateObject)
        catch e
            @warn "Failed to ensure AI mask backup: $e"
        end
    end

    # In-place continuous thick-line interpolation using KernelAbstractions
    StrokeRasterization.rasterize_polyline!(twoDimDat.dat, pointsToRasterize, strokeW, toSet)

    # ── Mark expert edit & set origin to EXPERT_CORRECTION ──
    if lesion_id > 0
        try
            MEH.mark_expert_edit!(lesion_id)
        catch e
            @warn "Failed to mark expert edit: $e"
        end
    end

    @debug "[PAINT-DBG] panel=$(mainStates[1].switchIndex) tex='$(texture.name)' val=$toSet pts=$(length(pointsToRasterize))"

    # Mark slice changed so consumer loop uploads dirty texture to GPU
    stateObject.isSliceChanged = true

    # Synchronize dirty flag on other visible panels
    for s in mainStates
        if s !== stateObject && sum(abs.(s.calcDimsStruct.mainImageQuadVert)) > 0.01f0
            s.isSliceChanged = true
        end
    end

    # Invalidate SUV/volume/centroid caches, async-recompute metrics, and mark mask dirty for auto-save
    try
        MEH.mark_tp_mask_dirty!(MEH.current_tp_index[])
        paint_id = round(Int, stateObject.valueForMasToSet.value)
        edit_id = paint_id > 0 ? paint_id : MEH.current_active_lesion_id[]
        if edit_id > 0 || paint_id == 0
            # Sync painted slice to tp_data_cache so organ mapping can find new voxels
            tp_idx = MEH.current_tp_index[]
            cur_slice = stateObject.currentDisplayedSlice
            entry = lock(MEH._tp_cache_lock) do
                haskey(MEH.tp_data_cache, tp_idx) ? MEH.tp_data_cache[tp_idx] : nothing
            end
            if entry !== nothing
                try
                    entry.mask_i16[:, :, cur_slice] .= Int16.(twoDimDat.dat)
                    if entry.mask isa Array{Int8, 3}
                        entry.mask[:, :, cur_slice] .= clamp.(twoDimDat.dat, Int8(-128), Int8(127))
                    else
                        entry.mask[:, :, cur_slice] .= twoDimDat.dat
                    end
                catch; end
            end
            if edit_id > 0
                MEH.invalidate_and_recompute_lesion_metrics_async!(edit_id, MEH.current_tp_index[])
            end
        end
    catch e
        # Log errors instead of silently swallowing
        println("[PAINT-ERR] SUV/organ mapping failed: $e"); flush(stdout)
    end
end#react_to_draw

# ── Measurement Helper Functions ──────────────────────────────────────────────

function _compute_sphere_suv!(m, mainStates)
    axialState = mainStates[1]
    pet_dat = nothing
    for dat in axialState.onScrollData.dataToScroll
        if dat.name == "PET"
            pet_dat = dat.dat
            break
        end
    end
    if pet_dat !== nothing
        cx_v, cy_v, cz_v = m.center_idx
        R = m.radius_mm
        sp = axialState.spacingsValue[1]
        sx, sy, sz = Float32(sp[1]), Float32(sp[2]), Float32(sp[3])
        if sz == 0.0f0; sz = 1.0f0; end
        rx = ceil(Int, R / sx)
        ry = ceil(Int, R / sy)
        rz = ceil(Int, R / sz)
        cxi, cyi, czi = round(Int, cx_v), round(Int, cy_v), round(Int, cz_v)

        sum_suv = 0.0f0
        max_suv = 0.0f0
        count = 0

        R2 = R^2
        for z in max(1, czi-rz):min(size(pet_dat, 3), czi+rz)
            dz = (z - czi) * sz
            dz2 = dz^2
            for y in max(1, cyi-ry):min(size(pet_dat, 2), cyi+ry)
                dy = (y - cyi) * sy
                dy2 = dy^2
                for x in max(1, cxi-rx):min(size(pet_dat, 1), cxi+rx)
                    dx = (x - cxi) * sx
                    dx2 = dx^2
                    if dx2 + dy2 + dz2 <= R2
                        val = pet_dat[x, y, z]
                        sum_suv += val
                        max_suv = max(max_suv, val)
                        count += 1
                    end
                end
            end
        end
        if count > 0
            m.suv_mean = sum_suv / count
            m.suv_max = max_suv
        else
            m.suv_mean = 0.0f0
            m.suv_max = 0.0f0
        end
    end
end

function _compute_line_suv!(m, mainStates)
    axialState = mainStates[1]
    pet_dat = nothing
    for dat in axialState.onScrollData.dataToScroll
        if dat.name == "PET"
            pet_dat = dat.dat
            break
        end
    end
    if pet_dat !== nothing
        sp = axialState.spacingsValue[1]
        sx, sy, sz = Float32(sp[1]), Float32(sp[2]), Float32(sp[3])

        dx_mm = (m.end_idx[1] - m.start_idx[1]) * sx
        dy_mm = (m.end_idx[2] - m.start_idx[2]) * sy
        dz_mm = (m.end_idx[3] - m.start_idx[3]) * sz
        len = sqrt(dx_mm^2 + dy_mm^2 + dz_mm^2)

        if len == 0
            cx, cy, cz = round(Int, m.start_idx[1]), round(Int, m.start_idx[2]), round(Int, m.start_idx[3])
            if checkbounds(Bool, pet_dat, cx, cy, cz)
                val = pet_dat[cx, cy, cz]
                m.suv_mean = val
                m.suv_max = val
            end
            return
        end

        steps = max(2, ceil(Int, len / min(sx, sy, sz) * 2))
        sum_suv = 0.0f0
        max_suv = 0.0f0
        count = 0

        for i in 0:steps
            t = i / steps
            cx = round(Int, m.start_idx[1] + t * (m.end_idx[1] - m.start_idx[1]))
            cy = round(Int, m.start_idx[2] + t * (m.end_idx[2] - m.start_idx[2]))
            cz = round(Int, m.start_idx[3] + t * (m.end_idx[3] - m.start_idx[3]))
            if checkbounds(Bool, pet_dat, cx, cy, cz)
                val = pet_dat[cx, cy, cz]
                sum_suv += val
                max_suv = max(max_suv, val)
                count += 1
            end
        end
        if count > 0
            m.suv_mean = sum_suv / count
            m.suv_max = max_suv
        else
            m.suv_mean = 0.0f0
            m.suv_max = 0.0f0
        end
    end
end

# ─── trigger_obs helper ──────────────────────────────────────────────────────
# For :obs_refresh_measurements — pass the display-object so the GUI can rebuild.
# For :obs_update_measurements — increment (existing integer Observable).
function _trigger_meas_obs(MEH, name::Symbol, obj_for_refresh)
    try
        LMW = MEH._get_lmw()
        if LMW !== nothing
            obs = getfield(LMW, :_lmw_observables)
            if haskey(obs, name)
                if name == :obs_refresh_measurements
                    obs[name][] = obj_for_refresh  # set to display-obj (triggers rebuild)
                    # Mark measurements dirty for HDF5 autosave
                    try MEH.mark_measurements_dirty!() catch; end
                else
                    obs[name][] += 1  # integer increment
                    # Also mark dirty on update (drag edits)
                    try MEH.mark_measurements_dirty!() catch; end
                end
            end
        end
    catch; end
end

# ─── Live info on top panel ──────────────────────────────────────────────────
function _update_cursor_info_sphere!(m, MEH)
    try
        info = "⬤ Sphere: SUV Mean=$(round(m.suv_mean, digits=2)) Max=$(round(m.suv_max, digits=2)) R=$(round(m.radius_mm, digits=1))mm"
        MEH.measurement_info_text[] = info
    catch; end
end

function _update_cursor_info_line!(lm, MEH)
    try
        len_cm = lm.length_mm / 10.0f0
        if lm.length_mm >= 10.0f0
            info = "━ Line: $(round(len_cm, digits=2))cm ($(round(lm.length_mm, digits=1))mm) | SUV Mean=$(round(lm.suv_mean, digits=2)) Max=$(round(lm.suv_max, digits=2))"
        else
            info = "━ Line: $(round(lm.length_mm, digits=1))mm | SUV Mean=$(round(lm.suv_mean, digits=2)) Max=$(round(lm.suv_max, digits=2))"
        end
        MEH.measurement_info_text[] = info
    catch; end
end

# ─── Hit-testing helpers ─────────────────────────────────────────────────────
function _find_nearby_sphere_edge(obj, point3d, mainStates, panel_id)
    # Check if the click is near the edge of an existing sphere
    hit_tolerance_mm = 5.0f0  # tolerance in mm
    axialState = mainStates[1]
    sp = axialState.spacingsValue[1]
    sx, sy, sz = Float32(sp[1]), Float32(sp[2]), Float32(sp[3])
    
    MEH_edit = parentmodule(@__MODULE__).SegmentationDisplay.MakieEventHandlers
    edit_id = MEH_edit.editing_measurement_id[]
    
    for (idx, m) in enumerate(obj.measurements)
        if m.is_active || m.id < 1; continue; end
        cx, cy, cz = m.center_idx
        px, py, pz = point3d
        
        # Distance from click to sphere center in mm
        dx_mm = (px - cx) * sx
        dy_mm = (py - cy) * sy
        dz_mm = (pz - cz) * sz
        dist_mm = sqrt(dx_mm^2 + dy_mm^2 + dz_mm^2)
        
        # Check if click is near the edge of the sphere
        R = m.radius_mm
        # When this is the "editing" measurement, be more generous — any click within 2x radius
        tolerance = (edit_id == m.id) ? R * 1.5f0 : hit_tolerance_mm
        min_dist = (edit_id == m.id) ? 0.0f0 : R * 0.3f0
        
        if abs(dist_mm - R) < tolerance && dist_mm > min_dist
            # Determine which axis the user is closest to
            abs_dx = abs(dx_mm)
            abs_dy = abs(dy_mm)
            abs_dz = abs(dz_mm)
            
            panel_mapped = panel_id > 5 ? panel_id - 5 : panel_id
            if panel_mapped == 1 || panel_mapped == 2 || panel_mapped == 5
                axis = abs_dx > abs_dy ? :x : :y
            elseif panel_mapped == 3
                axis = abs_dz > abs_dy ? :z : :y
            else
                axis = abs_dx > abs_dz ? :x : :z
            end
            return (idx, axis)
        end
    end
    return nothing
end

function _find_nearby_line_endpoint(obj, point3d, mainStates)
    hit_tolerance_vox = 8.0f0  # voxel distance tolerance
    px, py, pz = point3d
    
    MEH_edit = parentmodule(@__MODULE__).SegmentationDisplay.MakieEventHandlers
    edit_id = MEH_edit.editing_measurement_id[]
    
    for (idx, lm) in enumerate(obj.line_measurements)
        if lm.is_active || lm.id < 1; continue; end
        # Check start point
        ds = sqrt((px - lm.start_idx[1])^2 + (py - lm.start_idx[2])^2 + (pz - lm.start_idx[3])^2)
        de = sqrt((px - lm.end_idx[1])^2 + (py - lm.end_idx[2])^2 + (pz - lm.end_idx[3])^2)
        
        # More generous tolerance when this is the editing measurement
        tol = (edit_id == lm.id) ? hit_tolerance_vox * 3.0f0 : hit_tolerance_vox
        
        if ds <= tol && ds <= de
            return (idx, :start)
        elseif de <= tol
            return (idx, :end)
        end
    end
    return nothing
end

# ─── Main Sphere Handler ─────────────────────────────────────────────────────
function _handle_sphere_measurement(mousestr, mainStates, obj, center_idx, MEH, MeasMod)
    radius_mm = MEH.active_measurement_radius_mm[]
    active_idx = findfirst(m -> m.is_active && m.id > 0, obj.measurements)  # Exclude ghost (id=-1)
    # Also check if we're editing an edge
    editing_idx = findfirst(m -> m.editing_axis != :none && m.id > 0, obj.measurements)

    if mousestr.isLeftButtonDown
        if editing_idx !== nothing
            # Edge-editing mode: adjust radius on the editing axis
            m = obj.measurements[editing_idx]
            cx, cy, cz = m.center_idx
            px, py, pz = center_idx
            sp = mainStates[1].spacingsValue[1]
            sx, sy, sz = Float32(sp[1]), Float32(sp[2]), Float32(sp[3])
            
            if m.editing_axis == :x
                new_r = abs(px - cx) * sx
                m.radius_x_mm = max(2.0f0, new_r)
            elseif m.editing_axis == :y
                new_r = abs(py - cy) * sy
                m.radius_y_mm = max(2.0f0, new_r)
            elseif m.editing_axis == :z
                new_r = abs(pz - cz) * sz
                m.radius_z_mm = max(2.0f0, new_r)
            end
            _compute_sphere_suv!(m, mainStates)
            _update_cursor_info_sphere!(m, MEH)
            
        elseif active_idx === nothing
            # Check if clicking near an existing sphere's edge (edit mode)
            hit = _find_nearby_sphere_edge(obj, center_idx, mainStates, mainStates[1].switchIndex)
            if hit !== nothing
                sidx, axis = hit
                m = obj.measurements[sidx]
                m.editing_axis = axis
                # Initialize per-axis radii from base radius if not set
                if m.radius_x_mm <= 0; m.radius_x_mm = m.radius_mm; end
                if m.radius_y_mm <= 0; m.radius_y_mm = m.radius_mm; end
                if m.radius_z_mm <= 0; m.radius_z_mm = m.radius_mm; end
                _update_cursor_info_sphere!(m, MEH)
            else
                # Check if we clicked far away while in edit mode
                if MEH.editing_measurement_id[] > 0
                    MEH.editing_measurement_id[] = 0
                    MEH.editing_measurement_type[] = :none
                    return
                end
                # First click: convert ghost (id=-1) to real, or create new
                ghost_idx = findfirst(m -> m.id == -1, obj.measurements)
                new_meas_idx = 0
                if ghost_idx !== nothing
                    # Convert ghost to real measurement
                    m = obj.measurements[ghost_idx]
                    m.id = length(filter(mm -> mm.id > 0, obj.measurements)) + 1
                    m.center_idx = center_idx
                    m.radius_mm = radius_mm
                    m.is_active = true
                    new_meas_idx = ghost_idx
                else
                    next_color = (length(obj.measurements) + length(obj.line_measurements)) % 8 + 1
                    push!(obj.measurements, MeasMod.SphereMeasurement(
                        id = length(obj.measurements) + 1,
                        center_idx = center_idx,
                        radius_mm = radius_mm,
                        suv_mean = 0.0f0,
                        suv_max = 0.0f0,
                        is_active = true,
                        color_idx = next_color
                    ))
                    new_meas_idx = length(obj.measurements)
                end
                _compute_sphere_suv!(obj.measurements[new_meas_idx], mainStates)
                _update_cursor_info_sphere!(obj.measurements[new_meas_idx], MEH)
                _trigger_meas_obs(MEH, :obs_refresh_measurements, obj)
            end
        else
            # Dragging: update position and compute SUV live
            m = obj.measurements[active_idx]
            m.center_idx = center_idx
            m.radius_mm = radius_mm
            _compute_sphere_suv!(m, mainStates)
            _update_cursor_info_sphere!(m, MEH)
        end
    else
        # Mouse released: finalize
        if editing_idx !== nothing
            m = obj.measurements[editing_idx]
            m.editing_axis = :none
            _compute_sphere_suv!(m, mainStates)
            MEH.editing_measurement_id[] = m.id
            MEH.editing_measurement_type[] = :sphere
            _trigger_meas_obs(MEH, :obs_update_measurements, obj)
            _trigger_meas_obs(MEH, :obs_refresh_measurements, obj)
        elseif active_idx !== nothing
            m = obj.measurements[active_idx]
            m.center_idx = center_idx
            m.radius_mm = radius_mm
            m.is_active = false
            _compute_sphere_suv!(m, mainStates)
            # After creation, mark for editing so next click edits this sphere
            MEH.editing_measurement_id[] = m.id
            MEH.editing_measurement_type[] = :sphere
            # Initialize per-axis radii
            if m.radius_x_mm <= 0; m.radius_x_mm = m.radius_mm; end
            if m.radius_y_mm <= 0; m.radius_y_mm = m.radius_mm; end
            if m.radius_z_mm <= 0; m.radius_z_mm = m.radius_mm; end
            _trigger_meas_obs(MEH, :obs_update_measurements, obj)
            _trigger_meas_obs(MEH, :obs_refresh_measurements, obj)
        end
    end
end

# ─── Main Line Handler ───────────────────────────────────────────────────────
function _handle_line_measurement(mousestr, mainStates, obj, point3d, MEH, MeasMod)
    active_idx = findfirst(m -> m.is_active && m.id > 0, obj.line_measurements)  # Exclude ghost
    editing_idx = findfirst(m -> m.editing_endpoint != :none && m.id > 0, obj.line_measurements)

    function _compute_length!(lm)
        sp = mainStates[1].spacingsValue[1]
        sx, sy, sz = Float32(sp[1]), Float32(sp[2]), Float32(sp[3])
        dx_mm = (lm.end_idx[1] - lm.start_idx[1]) * sx
        dy_mm = (lm.end_idx[2] - lm.start_idx[2]) * sy
        dz_mm = (lm.end_idx[3] - lm.start_idx[3]) * sz
        lm.length_mm = sqrt(dx_mm^2 + dy_mm^2 + dz_mm^2)
    end

    if mousestr.isLeftButtonDown
        if editing_idx !== nothing
            # Endpoint-editing mode
            lm = obj.line_measurements[editing_idx]
            if lm.editing_endpoint == :start
                lm.start_idx = point3d
            else
                lm.end_idx = point3d
            end
            _compute_length!(lm)
            _compute_line_suv!(lm, mainStates)
            _update_cursor_info_line!(lm, MEH)
            
        elseif active_idx === nothing
            # Check if clicking near an existing line's endpoint (edit mode)
            hit = _find_nearby_line_endpoint(obj, point3d, mainStates)
            if hit !== nothing
                lidx, endpoint = hit
                lm = obj.line_measurements[lidx]
                lm.editing_endpoint = endpoint
                _update_cursor_info_line!(lm, MEH)
            else
                # Check if we clicked far away while in edit mode
                if MEH.editing_measurement_id[] > 0
                    MEH.editing_measurement_id[] = 0
                    MEH.editing_measurement_type[] = :none
                    return
                end
                # Start new line: convert ghost (id=-1) to real, or create new
                ghost_idx = findfirst(m -> m.id == -1, obj.line_measurements)
                new_line_idx = 0
                if ghost_idx !== nothing
                    # Convert ghost to real
                    lm = obj.line_measurements[ghost_idx]
                    lm.id = length(filter(mm -> mm.id > 0, obj.line_measurements)) + 1
                    lm.start_idx = point3d
                    lm.end_idx = point3d
                    lm.is_active = true
                    new_line_idx = ghost_idx
                else
                    next_color = (length(obj.measurements) + length(obj.line_measurements)) % 8 + 1
                    push!(obj.line_measurements, MeasMod.LineMeasurement(
                        id = length(obj.line_measurements) + 1,
                        start_idx = point3d,
                        end_idx = point3d,
                        length_mm = 0.0f0,
                        suv_mean = 0.0f0,
                        suv_max = 0.0f0,
                        is_active = true,
                        color_idx = next_color
                    ))
                    new_line_idx = length(obj.line_measurements)
                end
                _compute_line_suv!(obj.line_measurements[new_line_idx], mainStates)
                _update_cursor_info_line!(obj.line_measurements[new_line_idx], MEH)
                _trigger_meas_obs(MEH, :obs_refresh_measurements, obj)
            end
        else
            # Dragging: update endpoint
            lm = obj.line_measurements[active_idx]
            lm.end_idx = point3d
            _compute_length!(lm)
            _compute_line_suv!(lm, mainStates)
            _update_cursor_info_line!(lm, MEH)
        end
    else
        # Mouse released: finalize
        if editing_idx !== nothing
            lm = obj.line_measurements[editing_idx]
            lm.editing_endpoint = :none
            _compute_length!(lm)
            _compute_line_suv!(lm, mainStates)
            MEH.editing_measurement_id[] = lm.id
            MEH.editing_measurement_type[] = :line
            _trigger_meas_obs(MEH, :obs_update_measurements, obj)
            _trigger_meas_obs(MEH, :obs_refresh_measurements, obj)
        elseif active_idx !== nothing
            lm = obj.line_measurements[active_idx]
            lm.end_idx = point3d
            lm.is_active = false
            _compute_length!(lm)
            _compute_line_suv!(lm, mainStates)
            # After creation, mark for editing so next click edits this line
            MEH.editing_measurement_id[] = lm.id
            MEH.editing_measurement_type[] = :line
            _trigger_meas_obs(MEH, :obs_update_measurements, obj)
            _trigger_meas_obs(MEH, :obs_refresh_measurements, obj)
        end
    end
end

# ── End Measurement Helper Functions ──────────────────────────────────────────

"""
we use mouse coordinate to modify the texture that is currently active for modifications
    - we take information about texture currently active for modifications from variables stored in actor
    from texture specification we take also its id and its properties ...
"""
function reactToMouseDrag(mousestr::MouseStruct, mainStates::Vector{StateDataFields})
    mainState = mainStates[1] #first State for holding switch index information
    # obj = mainState.mainForDisplayObjects
    # textureList = mainState.textureToModifyVec
    mouseCoords = mousestr.lastCoordinates
    
    # 1. Update switchIndex based on mouse position — needs coords
    activeDragPanel = findfirst(s -> !isempty(s.lastPanDragCoords), mainStates)
    if activeDragPanel !== nothing && mousestr.isRightButtonDown
        mainState.switchIndex = activeDragPanel
    elseif !isempty(mouseCoords)
        win_idx = clamp(mousestr.window_id, 1, 2)
        zoomState = quadZoomStates[win_idx]
        if zoomState.isZoomed
            # When zoomed, always target the zoomed panel
            mainState.switchIndex = zoomState.zoomedPanel
        else
            viewportW = Float64(mainStates[1].calcDimsStruct.windowWidth)
            viewportH = Float64(mainStates[1].calcDimsStruct.windowHeight)
            actualW = mousestr.actualWindowWidth > 0 ? Float64(mousestr.actualWindowWidth) : viewportW
            actualH = mousestr.actualWindowHeight > 0 ? Float64(mousestr.actualWindowHeight) : viewportH
            x, y = (mouseCoords[1][1], mouseCoords[1][2])
            
            mainState.switchIndex = _detect_clicked_panel(x, y, actualW, actualH, mousestr.window_id, mainStates)
        end
    end # end !isempty(mouseCoords) for panel detection
    
    # If the right mouse button is released, clear the pan drag state
    if !mousestr.isRightButtonDown
        for state in mainStates
            empty!(state.lastPanDragCoords)
        end
    end

    # --- MEASUREMENT INTERCEPT (LEFT BUTTON DOWN - FALLBACK) ---
    # This handles the case where mouse events come as single MouseStruct 
    # (e.g., when is_painting_active is false and aggregation didn't trigger)
    if mousestr.isLeftButtonDown && !isempty(mouseCoords)
        MEH_fb = parentmodule(@__MODULE__).SegmentationDisplay.MakieEventHandlers
        if MEH_fb.measurements_mode[]
            sub_mode_fb = MEH_fb.measurement_sub_mode[]
            if sub_mode_fb == :sphere || sub_mode_fb == :line
                clickedPanel_fb = mainState.switchIndex
                if clickedPanel_fb >= 1 && clickedPanel_fb <= length(mainStates)
                    stObj = mainStates[clickedPanel_fb]
                    MeasMod_fb = parentmodule(@__MODULE__).Measurements
                    calcDim_fb = stObj.calcDimsStruct
                    actualW_fb = mousestr.actualWindowWidth > 0 ? Float64(mousestr.actualWindowWidth) : Float64(calcDim_fb.windowWidth)
                    actualH_fb = mousestr.actualWindowHeight > 0 ? Float64(mousestr.actualWindowHeight) : Float64(calcDim_fb.windowHeight)
                    
                    c_fb = mouseCoords[1]
                    texX_fb, texY_fb = StructsManag.getTextureCoordinatesFromScreen(c_fb[1], c_fb[2], calcDim_fb, actualW_fb, actualH_fb)
                    ix_fb, iy_fb = round(Int, texX_fb), round(Int, texY_fb)
                    slice_fb = stObj.currentDisplayedSlice
                    
                    cp_mapped_fb = clickedPanel_fb > 5 ? clickedPanel_fb - 5 : clickedPanel_fb
                    if cp_mapped_fb == 3
                        origX_fb, origY_fb, origZ_fb = slice_fb, ix_fb, iy_fb
                    elseif cp_mapped_fb == 4
                        origX_fb, origY_fb, origZ_fb = ix_fb, slice_fb, iy_fb
                    else
                        origX_fb, origY_fb, origZ_fb = ix_fb, iy_fb, slice_fb
                    end
                    
                    meas_obj_fb = mainStates[1].mainForDisplayObjects
                    if sub_mode_fb == :sphere
                        _handle_sphere_measurement(mousestr, mainStates, meas_obj_fb, (Float32(origX_fb), Float32(origY_fb), Float32(origZ_fb)), MEH_fb, MeasMod_fb)
                    elseif sub_mode_fb == :line
                        _handle_line_measurement(mousestr, mainStates, meas_obj_fb, (Float32(origX_fb), Float32(origY_fb), Float32(origZ_fb)), MEH_fb, MeasMod_fb)
                    end
                end
            end
        end
    end
    # --- END MEASUREMENT INTERCEPT (LEFT BUTTON DOWN) ---

    # If the left mouse button is released, clear the paint stroke tail
    if !mousestr.isLeftButtonDown
        for state in mainStates
            empty!(state.lastPaintCoords)
        end
        
        # --- MEASUREMENT INTERCEPT (RELEASE) ---
        MEH = parentmodule(@__MODULE__).SegmentationDisplay.MakieEventHandlers
        sub_mode = MEH.measurement_sub_mode[]
        if MEH.measurements_mode[] && (sub_mode == :sphere || sub_mode == :line)
            clickedPanel = mainState.switchIndex
            if clickedPanel >= 1 && clickedPanel <= length(mainStates)
                stateObject = mainStates[clickedPanel]
                MeasMod = parentmodule(@__MODULE__).Measurements
                calcDim = stateObject.calcDimsStruct
                actualW = mousestr.actualWindowWidth > 0 ? Float64(mousestr.actualWindowWidth) : Float64(calcDim.windowWidth)
                actualH = mousestr.actualWindowHeight > 0 ? Float64(mousestr.actualWindowHeight) : Float64(calcDim.windowHeight)
                
                if !isempty(mouseCoords)
                    c = mouseCoords[1]
                    texX, texY = StructsManag.getTextureCoordinatesFromScreen(c[1], c[2], calcDim, actualW, actualH)
                    ix, iy = round(Int, texX), round(Int, texY)
                    slice = stateObject.currentDisplayedSlice
                    
                    cp_mapped = clickedPanel > 5 ? clickedPanel - 5 : clickedPanel
                    if cp_mapped == 3 # Sagittal
                        origX, origY, origZ = slice, ix, iy
                    elseif cp_mapped == 4 # Coronal
                        origX, origY, origZ = ix, slice, iy
                    else # Axial
                        origX, origY, origZ = ix, iy, slice
                    end
                    
                    meas_obj_rel = mainStates[1].mainForDisplayObjects
                    if sub_mode == :sphere
                        _handle_sphere_measurement(mousestr, mainStates, meas_obj_rel, (Float32(origX), Float32(origY), Float32(origZ)), MEH, MeasMod)
                    elseif sub_mode == :line
                        _handle_line_measurement(mousestr, mainStates, meas_obj_rel, (Float32(origX), Float32(origY), Float32(origZ)), MEH, MeasMod)
                    end
                end
            end
        end
        # --- END MEASUREMENT INTERCEPT ---
    end

    # 2. Right-click cross-plane jumping or panning
    if !isempty(mouseCoords) && mousestr.isRightButtonDown && !isempty(mainStates)
        viewportW = Float64(mainStates[1].calcDimsStruct.windowWidth)
        viewportH = Float64(mainStates[1].calcDimsStruct.windowHeight)
        actualW = mousestr.actualWindowWidth > 0 ? Float64(mousestr.actualWindowWidth) : viewportW
        actualH = mousestr.actualWindowHeight > 0 ? Float64(mousestr.actualWindowHeight) : viewportH
        x, y = (mouseCoords[1][1], mouseCoords[1][2])
        clickedPanel = mainState.switchIndex
        
        panelState = mainStates[clickedPanel]
        
        if isempty(panelState.lastPanDragCoords)
            # Read actual rendered vertex positions from the panel's calcDimsStruct
            texX, texY = getTextureCoordinatesFromScreen(x, y, panelState.calcDimsStruct, actualW, actualH)
            
            if panelState.moveLesionMode
                target_id = 0
                for ts in panelState.mainForDisplayObjects.listOfTextSpecifications
                    if (ts.isMultiDiscreteMask || ts.name == "Mask" || ts.name == "manualModif") && ts.name != "Anatomy" && !isempty(ts.minAndMaxValue) && ts.minAndMaxValue[1] > 0
                        target_id = Int(round(ts.minAndMaxValue[1]))
                        break
                    end
                end
                if target_id <= 0
                    target_id = panelState.valueForMasToSet.value
                end
                if target_id <= 0
                    for ts in mainStates[1].mainForDisplayObjects.listOfTextSpecifications
                        if (ts.isMultiDiscreteMask || ts.name == "Mask" || ts.name == "manualModif") && ts.name != "Anatomy" && !isempty(ts.minAndMaxValue) && ts.minAndMaxValue[1] > 0
                            target_id = Int(round(ts.minAndMaxValue[1]))
                            break
                        end
                    end
                end
                if target_id <= 0
                    target_id = 1
                end
                
                panelState.movingLesionID = target_id
                panelState.movingLesionStartTex = (texX, texY)
                panelState.movingLesionLastDelta = CartesianIndex(0,0,0)
                panelState.movingLesionSourceName = ""
                seg_vol = nothing
                # In compare mode, search the right panel's data for the lesion
                MEH = parentmodule(@__MODULE__).SegmentationDisplay.MakieEventHandlers
                search_panel = (MEH.compare_mode[] && clickedPanel == 5) ? mainStates[5] : mainStates[1]
                for dat in search_panel.onScrollData.dataToScroll
                    if dat.name == "Mask" || dat.name == "segmentation"
                        T_elem = eltype(dat.dat)
                        if panelState.movingLesionID > 0 && any(dat.dat .== round(T_elem, panelState.movingLesionID))
                            seg_vol = dat.dat
                            panelState.movingLesionSourceName = dat.name
                            break
                        elseif seg_vol === nothing
                            seg_vol = dat.dat
                            panelState.movingLesionSourceName = dat.name
                        end
                    elseif dat.name == "manualModif" && seg_vol === nothing
                        seg_vol = dat.dat
                        panelState.movingLesionSourceName = dat.name
                    end
                end
                
                if seg_vol !== nothing && panelState.movingLesionID > 0
                    try
                        MEH = parentmodule(@__MODULE__).SegmentationDisplay.MakieEventHandlers
                        tp_idx = (MEH.compare_mode[] && clickedPanel == 5) ? MEH.compare_right_tp[] : MEH.current_tp_index[]
                        MEH.ensure_ai_mask_backup!(panelState.movingLesionID, tp_idx, panelState)
                    catch; end
                    T_elem = eltype(seg_vol)
                    panelState.movingLesionOriginalCoords = findall(seg_vol .== round(T_elem, panelState.movingLesionID))
                    panelState.movingLesionOriginalBGs = zeros(T_elem, length(panelState.movingLesionOriginalCoords))
                    @info "Move Lesion START: lesion=$(panelState.movingLesionID) found $(length(panelState.movingLesionOriginalCoords)) voxels"
                else
                    panelState.movingLesionOriginalCoords = CartesianIndex{3}[]
                    panelState.movingLesionOriginalBGs = Float32[]
                    @warn "Move Lesion START: No voxels found for lesion $(panelState.movingLesionID)"
                end
                panelState.lastPanDragCoords = [CartesianIndex(Int(round(x)), Int(round(y)))]
                return
            end
            
            # Initial press: Do the jump!
            panelState.lastPanDragCoords = [CartesianIndex(Int(round(x)), Int(round(y)))]
            
            @info "RIGHT-CLICK: panel=$clickedPanel windowXY=($x,$y) viewport=$(Int(viewportW))x$(Int(viewportH)) actual=$(Int(actualW))x$(Int(actualH))"
            @info "  texX=$texX texY=$texY"
            
            currentSlice = panelState.currentDisplayedSlice
            
            clickedPanel_mapped = clickedPanel > 5 ? clickedPanel - 5 : clickedPanel
            if clickedPanel_mapped == 1 || clickedPanel_mapped == 2 || clickedPanel_mapped == 5
                origX, origY, origZ = texX, texY, currentSlice
            elseif clickedPanel_mapped == 3  # Sagittal
                origY, origZ, origX = texX, texY, currentSlice
            else # Bottom-Right (4) (Coronal)
                origX, origZ, origY = texX, texY, currentSlice
            end
            
            # Ensure lastRecordedMousePosition is updated for ALL panels so scroll sync knows the intersection!
            for i in 1:length(mainStates)
                i_mapped = i > 5 ? i - 5 : i
                if i_mapped == 1 || i_mapped == 2 || i_mapped == 5
                    mainStates[i].lastRecordedMousePosition = CartesianIndex(origX, origY, origZ)
                elseif i_mapped == 3
                    mainStates[i].lastRecordedMousePosition = CartesianIndex(origY, origZ, origX)
                else
                    mainStates[i].lastRecordedMousePosition = CartesianIndex(origX, origZ, origY)
                end
            end
            
            @info "  origX=$origX origY=$origY origZ=$origZ currentSlice=$currentSlice"
            @info "  Axial scrolls Z(1-$(mainStates[1].onScrollData.slicesNumber)) Sag scrolls origX(1-$(mainStates[3].onScrollData.slicesNumber)) Cor scrolls origY(1-$(mainStates[4].onScrollData.slicesNumber))"
            
            MEH = parentmodule(@__MODULE__).SegmentationDisplay.MakieEventHandlers
            
            # Jump other panels to the corresponding slices
            # Panel 1, 2 & 5 scroll through Z (origZ), Panel 3 scrolls through origX, Panel 4 scrolls through origY
            targets = [(1, origZ), (2, origZ), (3, origX), (4, origY)]
            if length(mainStates) >= 5
                push!(targets, (5, origZ))
            end
            if length(mainStates) >= 10 && MEH._m2_crosshair_sync[]
                push!(targets, (6, origZ), (7, origZ), (8, origX), (9, origY), (10, origZ))
            end
            
            for (p_idx, targetSlice) in targets
                if p_idx != clickedPanel && p_idx <= length(mainStates)
                    otherState = mainStates[p_idx]
                    
                    # Read max slices for clamp
                    lastSlice = otherState.onScrollData.slicesNumber
                    newSlice = clamp(targetSlice, 1, lastSlice)
                    
                    # Always synchronize data so that crosshairs stay locked
                    singleSlDat = otherState.onScrollData.dataToScroll |>
                        (scrDat) -> map(threeDimDat -> threeToTwoDimm(threeDimDat.type, Int64(newSlice), otherState.onScrollData.dimensionToScroll, threeDimDat), scrDat) |>
                        (twoDimList) -> SingleSliceDat(listOfDataAndImageNames=twoDimList, sliceNumber=newSlice, textToDisp=getTextForCurrentSlice(otherState.onScrollData, Int32(newSlice)))
                    
                    # Upload new texture data to GPU (without rendering/SwapBuffers)
                    for updateDat in singleSlDat.listOfDataAndImageNames
                        idx = findfirst(ts -> ts.name == updateDat.name, otherState.mainForDisplayObjects.listOfTextSpecifications)
                        if idx !== nothing
                            texSpec = otherState.mainForDisplayObjects.listOfTextSpecifications[idx]
                            # GPU zoom/pan: upload raw unzoomed data — zoom/pan applied by vertex shader
                            updateTexture(updateDat.type, updateDat.dat, texSpec, 0, 0, otherState.calcDimsStruct.imageTextureWidth, otherState.calcDimsStruct.imageTextureHeight)
                        end
                    end
                    
                    # Update state (slice number and display data)
                    otherState.currentlyDispDat = singleSlDat
                    otherState.currentDisplayedSlice = newSlice
                    otherState.isSliceChanged = true
                end
            end
        else
            texX, texY = getTextureCoordinatesFromScreen(x, y, panelState.calcDimsStruct, actualW, actualH)
            
            if panelState.moveLesionMode
                if !isempty(panelState.movingLesionOriginalCoords) && panelState.movingLesionID > 0
                    startTexX, startTexY = panelState.movingLesionStartTex
                    dx_tex = round(Int, texX - startTexX)
                    dy_tex = round(Int, texY - startTexY)
                    
                    dx_vox, dy_vox, dz_vox = 0, 0, 0
                    if clickedPanel == 1 || clickedPanel == 2 || clickedPanel == 5
                        dx_vox, dy_vox = dx_tex, dy_tex
                    elseif clickedPanel == 3 # Sagittal (2,3,1) -> x is slice, y is texX, z is texY
                        dy_vox, dz_vox = dx_tex, texY - startTexY # keeping logic aligned
                        # Simplified delta mapping for sagittal
                        dx_vox, dy_vox, dz_vox = 0, dx_tex, dy_tex
                    else # Coronal (1,3,2) -> y is slice, x is texX, z is texY
                        dx_vox, dy_vox, dz_vox = dx_tex, 0, dy_tex
                    end
                    
                    new_delta = CartesianIndex(dx_vox, dy_vox, dz_vox)
                    
                    if new_delta != panelState.movingLesionLastDelta
                        target_id = panelState.movingLesionID
                        orig_coords = panelState.movingLesionOriginalCoords
                        orig_bgs = panelState.movingLesionOriginalBGs
                        old_d = panelState.movingLesionLastDelta
                        
                        dx, dy, dz = new_delta[1], new_delta[2], new_delta[3]
                        old_dx, old_dy, old_dz = old_d[1], old_d[2], old_d[3]
                        
                        MEH = parentmodule(@__MODULE__).SegmentationDisplay.MakieEventHandlers
                        panels_to_update = (MEH.compare_mode[] && clickedPanel == 5) ? [5] : collect(1:length(mainStates))
                        
                        source_name = panelState.movingLesionSourceName
                        processed_arrays = Set{UInt}()  # track objectid to skip duplicate array refs
                        
                        for p_idx in panels_to_update
                            st = mainStates[p_idx]
                            p_delta = (p_idx == 3) ? CartesianIndex(dy, dz, dx) : ((p_idx == 4) ? CartesianIndex(dx, dz, dy) : CartesianIndex(dx, dy, dz))
                            p_old_d = (p_idx == 3) ? CartesianIndex(old_dy, old_dz, old_dx) : ((p_idx == 4) ? CartesianIndex(old_dx, old_dz, old_dy) : CartesianIndex(old_dx, old_dy, old_dz))
                            
                            p_orig_coords = map(orig_coords) do c
                                (p_idx == 3) ? CartesianIndex(c[2], c[3], c[1]) : ((p_idx == 4) ? CartesianIndex(c[1], c[3], c[2]) : c)
                            end
                            
                            for dat in st.onScrollData.dataToScroll
                                if dat.name == source_name
                                    seg_v = dat.dat
                                    arr_id = objectid(seg_v)
                                    if arr_id in processed_arrays
                                        continue  # skip duplicate array (panels 1+5 share same reference)
                                    end
                                    push!(processed_arrays, arr_id)
                                    
                                    # 1. Restore old
                                    last_coords = [c + p_old_d for c in p_orig_coords]
                                    for (j, c) in enumerate(last_coords)
                                        if checkbounds(Bool, seg_v, c) && j <= length(orig_bgs)
                                            seg_v[c] = orig_bgs[j]
                                        end
                                    end
                                    
                                    # 2. Write new
                                    new_coords = [c + p_delta for c in p_orig_coords]
                                    for c in new_coords
                                        if checkbounds(Bool, seg_v, c)
                                            seg_v[c] = round(eltype(seg_v), target_id)
                                        end
                                    end
                                end
                            end
                        end
                        
                        panelState.movingLesionLastDelta = new_delta
                        
                        # Synchronize tp_data_cache
                        try
                            MEH = parentmodule(@__MODULE__).SegmentationDisplay.MakieEventHandlers
                            tp_idx = (MEH.compare_mode[] && clickedPanel == 5) ? MEH.compare_right_tp[] : MEH.current_tp_index[]
                            entry = lock(MEH._tp_cache_lock) do
                                haskey(MEH.tp_data_cache, tp_idx) ? MEH.tp_data_cache[tp_idx] : nothing
                            end
                            if entry !== nothing
                                # The source_name should be "Mask" or "manualModif"
                                # We sync it from the canonical axial panel
                                if source_name == "Mask" || source_name == "manualModif"
                                    # Find the corresponding dataToScroll in the current panel
                                    for st_dat in mainStates[clickedPanel].onScrollData.dataToScroll
                                        if st_dat.name == source_name
                                            # We need to reshape/permute back to axial based on clickedPanel
                                            vol = if clickedPanel == 3 # Sagittal (Y, Z, X) -> back to (X, Y, Z)
                                                permutedims(st_dat.dat, (3, 1, 2))
                                            elseif clickedPanel == 4 # Coronal (X, Z, Y) -> back to (X, Y, Z)
                                                permutedims(st_dat.dat, (1, 3, 2))
                                            else # Axial (1, 2, 5)
                                                st_dat.dat
                                            end
                                            
                                            if entry.mask isa Array{Int8, 3}
                                                entry.mask .= round.(Int8, vol)
                                            else
                                                entry.mask .= round.(Int16, vol)
                                            end
                                            if entry.mask_i16 !== nothing
                                                entry.mask_i16 .= round.(Int16, vol)
                                            end
                                            MEH.mark_tp_mask_dirty!(tp_idx)
                                            MEH.mark_expert_edit!(panelState.movingLesionID)
                                            break
                                        end
                                    end
                                end
                            end
                        catch err
                            @warn "Failed to sync tp_data_cache on move lesion: $err"
                        end
                        
                        # Mark all visible panels dirty so consumer loop immediately uploads translated textures
                        for p in 1:length(mainStates)
                            if sum(abs.(mainStates[p].calcDimsStruct.mainImageQuadVert)) > 0.01f0
                                mainStates[p].isSliceChanged = true
                            end
                        end
                    end
                end
                return
            end
            
            # Dragging: Do the pan!
            lastX, lastY = panelState.lastPanDragCoords[1][1], panelState.lastPanDragCoords[1][2]
            
            dx = x - lastX
            dy = y - lastY  
            
            # Horizontal motion dx maps to panY (horizontal offset in shader pc.uvOffset.x)
            # Vertical motion dy maps to panX (vertical offset in shader pc.uvOffset.y)
            panSpeedHoriz = Float32(dx / actualW) / max(0.1f0, panelState.calcDimsStruct.zoom)
            panSpeedVert  = Float32(dy / actualH) / max(0.1f0, panelState.calcDimsStruct.zoom)
            
            # Drag right (dx > 0) -> panY decreases -> image shifts right (follows cursor)
            # Drag down  (dy > 0) -> panX increases -> image shifts down  (follows cursor)
            panelState.calcDimsStruct.panY = clamp(panelState.calcDimsStruct.panY - panSpeedHoriz, -1.0f0, 1.0f0)
            panelState.calcDimsStruct.panX = clamp(panelState.calcDimsStruct.panX + panSpeedVert, -1.0f0, 1.0f0)
            
            panelState.lastPanDragCoords = [CartesianIndex(Int(round(x)), Int(round(y)))]
            # GPU pan: no reactToScroll needed — render loop picks up new pan via setZoomPanUniforms
        end
    end # end right-click handler

    # ─── GHOST Measurement Preview (on EVERY mouse move) ────────────────────────
    # When measurement mode is active, show a preview sphere/line at the cursor
    # position BEFORE the user clicks. This gives live feedback as they move.
    try
        MEH_ghost = parentmodule(@__MODULE__).SegmentationDisplay.MakieEventHandlers
        if MEH_ghost.measurements_mode[] && !isempty(mouseCoords) && !mousestr.isLeftButtonDown
            sub_mode_g = MEH_ghost.measurement_sub_mode[]
            clickedPanel_g = mainState.switchIndex
            edit_id_g = MEH_ghost.editing_measurement_id[]
            
            # If we're in edit mode (just placed a measurement), show its info and suppress ghost
            if edit_id_g > 0 && clickedPanel_g >= 1 && clickedPanel_g <= length(mainStates)
                meas_obj_edit = mainStates[1].mainForDisplayObjects
                # Remove any existing ghosts
                if any(m -> m.id == -1, meas_obj_edit.measurements)
                    filter!(m -> m.id != -1, meas_obj_edit.measurements)
                end
                if any(m -> m.id == -1, meas_obj_edit.line_measurements)
                    filter!(m -> m.id != -1, meas_obj_edit.line_measurements)
                end
                # Show info for the measurement being edited
                if sub_mode_g == :sphere
                    eidx = findfirst(m -> m.id == edit_id_g, meas_obj_edit.measurements)
                    if eidx !== nothing
                        _update_cursor_info_sphere!(meas_obj_edit.measurements[eidx], MEH_ghost)
                    end
                elseif sub_mode_g == :line
                    eidx = findfirst(m -> m.id == edit_id_g, meas_obj_edit.line_measurements)
                    if eidx !== nothing
                        _update_cursor_info_line!(meas_obj_edit.line_measurements[eidx], MEH_ghost)
                    end
                end
            elseif clickedPanel_g >= 1 && clickedPanel_g <= length(mainStates) && (sub_mode_g == :sphere || sub_mode_g == :line)
                MeasMod_g = parentmodule(@__MODULE__).Measurements
                stObj_g = mainStates[clickedPanel_g]
                calcDim_g = stObj_g.calcDimsStruct
                actualW_g = mousestr.actualWindowWidth > 0 ? Float64(mousestr.actualWindowWidth) : Float64(calcDim_g.windowWidth)
                actualH_g = mousestr.actualWindowHeight > 0 ? Float64(mousestr.actualWindowHeight) : Float64(calcDim_g.windowHeight)
                
                c_g = mouseCoords[1]
                texX_g, texY_g = StructsManag.getTextureCoordinatesFromScreen(c_g[1], c_g[2], calcDim_g, actualW_g, actualH_g)
                ix_g, iy_g = round(Int, texX_g), round(Int, texY_g)
                slice_g = stObj_g.currentDisplayedSlice
                
                cp_mapped_g = clickedPanel_g > 5 ? clickedPanel_g - 5 : clickedPanel_g
                if cp_mapped_g == 3
                    origX_g, origY_g, origZ_g = slice_g, ix_g, iy_g
                elseif cp_mapped_g == 4
                    origX_g, origY_g, origZ_g = ix_g, slice_g, iy_g
                else
                    origX_g, origY_g, origZ_g = ix_g, iy_g, slice_g
                end
                
                meas_obj_g = mainStates[1].mainForDisplayObjects
                point3d_g = (Float32(origX_g), Float32(origY_g), Float32(origZ_g))
                
                if sub_mode_g == :sphere
                    # Clean up any line ghosts when in sphere mode
                    if any(m -> m.id == -1, meas_obj_g.line_measurements)
                        filter!(m -> m.id != -1, meas_obj_g.line_measurements)
                    end
                    # Update or create ghost sphere
                    ghost_idx = findfirst(m -> m.id == -1, meas_obj_g.measurements)
                    if ghost_idx === nothing
                        # Create ghost sphere (id=-1 marks it as ghost)
                        radius_g = MEH_ghost.active_measurement_radius_mm[]
                        push!(meas_obj_g.measurements, MeasMod_g.SphereMeasurement(
                            id = -1,
                            center_idx = point3d_g,
                            radius_mm = radius_g,
                            suv_mean = 0.0f0,
                            suv_max = 0.0f0,
                            is_active = true,
                            color_idx = (length(meas_obj_g.measurements) + length(meas_obj_g.line_measurements)) % 8 + 1
                        ))
                        ghost_idx = length(meas_obj_g.measurements)
                    end
                    ghost = meas_obj_g.measurements[ghost_idx]
                    ghost.center_idx = point3d_g
                    ghost.radius_mm = MEH_ghost.active_measurement_radius_mm[]
                    ghost.is_active = true
                    _compute_sphere_suv!(ghost, mainStates)
                    _update_cursor_info_sphere!(ghost, MEH_ghost)
                    
                elseif sub_mode_g == :line
                    # Clean up any sphere ghosts when in line mode
                    if any(m -> m.id == -1, meas_obj_g.measurements)
                        filter!(m -> m.id != -1, meas_obj_g.measurements)
                    end
                    # Update or create ghost line
                    ghost_idx = findfirst(m -> m.id == -1, meas_obj_g.line_measurements)
                    if ghost_idx === nothing
                        push!(meas_obj_g.line_measurements, MeasMod_g.LineMeasurement(
                            id = -1,
                            start_idx = point3d_g,
                            end_idx = point3d_g,
                            length_mm = 0.0f0,
                            suv_mean = 0.0f0,
                            suv_max = 0.0f0,
                            is_active = true,
                            color_idx = (length(meas_obj_g.measurements) + length(meas_obj_g.line_measurements)) % 8 + 1
                        ))
                        ghost_idx = length(meas_obj_g.line_measurements)
                    end
                    ghost_line = meas_obj_g.line_measurements[ghost_idx]
                    ghost_line.start_idx = point3d_g
                    ghost_line.end_idx = point3d_g
                    ghost_line.is_active = true
                    _compute_line_suv!(ghost_line, mainStates)
                    # Show position info for ghost line
                    MEH_ghost.measurement_info_text[] = "━ Line: move cursor, click to set start point"
                end
            end
        end
        # Clear ghost when measurement mode is off
        if !isempty(mainStates) && !MEH_ghost.measurements_mode[]
            meas_obj_clr = mainStates[1].mainForDisplayObjects
            if !isempty(meas_obj_clr.measurements) && any(m -> m.id == -1, meas_obj_clr.measurements)
                filter!(m -> m.id != -1, meas_obj_clr.measurements)
            end
            if !isempty(meas_obj_clr.line_measurements) && any(m -> m.id == -1, meas_obj_clr.line_measurements)
                filter!(m -> m.id != -1, meas_obj_clr.line_measurements)
            end
            if !isempty(MEH_ghost.measurement_info_text[])
                MEH_ghost.measurement_info_text[] = ""
            end
        end
    catch; end

    # ─── Cursor Info Readout (runs on OpenGL thread via on_next! dispatch) ───────
    # Updates the cursor_info_text Observable and GLFW window title with:
    # study name, HU, SUV, lesion name, view orientation, and slice number.
    # Coordinate mapping is orientation-aware (axial/sagittal/coronal).
    try
        MEH = parentmodule(@__MODULE__).SegmentationDisplay.MakieEventHandlers
        clickedPanel = mainState.switchIndex
        if clickedPanel >= 1 && clickedPanel <= length(mainStates) && !isempty(mouseCoords)
            panelState = mainStates[clickedPanel]
            viewportW = Float64(mainStates[1].calcDimsStruct.windowWidth)
            viewportH = Float64(mainStates[1].calcDimsStruct.windowHeight)
            actualW = mousestr.actualWindowWidth > 0 ? Float64(mousestr.actualWindowWidth) : viewportW
            actualH = mousestr.actualWindowHeight > 0 ? Float64(mousestr.actualWindowHeight) : viewportH
            x, y = mouseCoords[1][1], mouseCoords[1][2]
            texX, texY = getTextureCoordinatesFromScreen(x, y, panelState.calcDimsStruct, actualW, actualH)
            ix, iy = Int(round(texX)), Int(round(texY))
            currentSlice = panelState.currentDisplayedSlice

            # Build info parts
            parts = String[]

            # Study label (compare mode aware)
            tp_idx = MEH.current_tp_index[]
            tp_label = get(MEH.tp_labels, tp_idx, "TP $tp_idx")
            if MEH.compare_mode[]
                r_idx = MEH.compare_right_tp[]
                r_label = get(MEH.tp_labels, r_idx, "TP $r_idx")
                study_str = "L: $tp_label | R: $r_label"
            else
                study_str = tp_label
            end

            # Read values from 3D volumes using panel-local coordinates
            # (panels 3,4 have reoriented data, so ix,iy,currentSlice are already correct)
            for dat in panelState.onScrollData.dataToScroll
                if dat.name == "CT" && checkbounds(Bool, dat.dat, ix, iy, currentSlice)
                    hu = dat.dat[ix, iy, currentSlice]
                    push!(parts, "HU: $(round(Int, hu))")
                elseif dat.name == "PET" && checkbounds(Bool, dat.dat, ix, iy, currentSlice)
                    suv = dat.dat[ix, iy, currentSlice]
                    push!(parts, "SUV: $(round(suv, digits=2))")
                elseif (dat.name == "Mask" || dat.name == "segmentation") && checkbounds(Bool, dat.dat, ix, iy, currentSlice)
                    mask_val = dat.dat[ix, iy, currentSlice]
                    if mask_val > 0
                        lid = Int(round(mask_val))
                        # Look up lesion name: prioritize RTOG / clinical name from HDF5
                        organ = try
                            tp = MEH.current_tp_index[]
                            seg_names = get(MEH.tp_segment_names, tp, Dict{Int, String}())
                            s_name = get(seg_names, lid, "")
                            if isempty(s_name) && tp != 0
                                s_name = get(get(MEH.tp_segment_names, 0, Dict{Int, String}()), lid, "")
                            end
                            !isempty(s_name) ? s_name : get(MEH.global_organ_mapping[], lid, "")
                        catch
                            ""
                        end
                        label = isempty(organ) ? "Lesion $lid" : "$organ (L$lid)"
                        
                        # On MRI: also show the anatomy zone from max_anatomy under this voxel
                        try
                            tp = MEH.current_tp_index[]
                            panel_mod = uppercase(get(MEH.tp_modalities, tp, "PET"))
                            if panel_mod in ("T2", "MRI", "MR", "T1", "ADC", "DWI")
                                for adat in panelState.onScrollData.dataToScroll
                                    if adat.name == "Anatomy" && checkbounds(Bool, adat.dat, ix, iy, currentSlice)
                                        anat_val = Int(round(adat.dat[ix, iy, currentSlice]))
                                        if anat_val > 0
                                            labels = get(MEH.anatomy_labels_cache, tp, Dict{Int,String}())
                                            anat_lbl = get(labels, anat_val, get(MEH.global_ts_names[], anat_val, ""))
                                            if !isempty(anat_lbl)
                                                label = "$label in $anat_lbl"
                                            end
                                        end
                                        break
                                    end
                                end
                            end
                        catch; end
                        
                        push!(parts, label)
                    end
                elseif dat.name == "Anatomy" && checkbounds(Bool, dat.dat, ix, iy, currentSlice)
                    anat_val = Int(round(dat.dat[ix, iy, currentSlice]))
                    if anat_val > 0
                        # Look up anatomy label from per-TP cached names or global
                        anat_name = try
                            tp = MEH.current_tp_index[]
                            labels = get(MEH.anatomy_labels_cache, tp, Dict{Int,String}())
                            lbl = get(labels, anat_val, "")
                            if isempty(lbl) || occursin("class_", lbl)
                                # Fallback to global ts_names (actual organ names)
                                get(MEH.global_ts_names[], anat_val, "")
                            else
                                lbl
                            end
                        catch
                            ""
                        end
                        if !isempty(anat_name)
                            push!(parts, anat_name)
                        end
                    end
                end
            end

            # Panel orientation indicator
            view_name = if clickedPanel == 1 || clickedPanel == 2 || clickedPanel == 5
                "Ax"
            elseif clickedPanel == 3
                "Sag"
            else
                "Cor"
            end
            push!(parts, "[$view_name] Sl:$currentSlice")

            info_str = join(parts, " | ")
            MEH.cursor_info_text[] = info_str
            MEH.cursor_study_text[] = study_str

            # Track 3D voxel position (axial orientation) for anatomy lookups
            # Panels 1,2,5 are axial: (ix, iy, slice) maps to (x, y, z)
            # Panel 3 is sagittal: data is PermutedDimsArray(vol, (3,2,1)), so (ix,iy,slice)→(slice,iy,ix)
            # Panel 4 is coronal:  data is PermutedDimsArray(vol, (1,3,2)), so (ix,iy,slice)→(ix,slice,iy)
            vox = if clickedPanel == 3
                (currentSlice, iy, ix)
            elseif clickedPanel == 4
                (ix, currentSlice, iy)
            else
                (ix, iy, currentSlice)
            end
            MEH.current_viewer_position[] = vox

            # Update GLFW window title (already on OpenGL thread inside on_next!, safe to call directly)
            meas_info = MEH.measurement_info_text[]
            title = isempty(meas_info) ? "MedEye3d - $study_str | $info_str" : "MedEye3d - $study_str | $info_str | $meas_info"
            GLFW.SetWindowTitle(panelState.mainForDisplayObjects.window, title)
        end
    catch
        # Never let info readout crash the mouse handler
    end

end#..ReactToScroll


"""
Handles double-click panel zoom toggle in QuadImage mode.
Dispatched via on_next!(states, data::DoubleClickEvent) — same pattern as all other event types.
"""
function reactToDoubleClick(event::DoubleClickEvent, mainStates::Vector{StateDataFields})
    if length(mainStates) < 4
        return
    end

    # Guard: Disable double click zoom while painting or erasing is active
    if !isempty(mainStates) && isdefined(mainStates[1], :valueForMasToSet) && mainStates[1].valueForMasToSet.is_painting_active
        @info "Double-click zoom ignored: painting/erasing is currently active"
        return
    end

    win_idx = clamp(event.window_id, 1, 2)
    zoomState = quadZoomStates[win_idx]
    is_m2 = (event.window_id == 2)
    panel_range = (is_m2 && length(mainStates) >= 10) ? (6:10) : (1:min(5, length(mainStates)))

    # Determine which panel was clicked from cursor position
    viewportW = Float64(mainStates[panel_range.start].calcDimsStruct.windowWidth)
    viewportH = Float64(mainStates[panel_range.start].calcDimsStruct.windowHeight)
    actualW = event.actualWindowWidth > 0 ? Float64(event.actualWindowWidth) : viewportW
    actualH = event.actualWindowHeight > 0 ? Float64(event.actualWindowHeight) : viewportH

    clickedPanel = if zoomState.isZoomed
        zoomState.zoomedPanel  # when zoomed, always target the zoomed panel
    else
        _detect_clicked_panel(event.x, event.y, actualW, actualH, event.window_id, mainStates)
    end

    if !zoomState.isZoomed
        @info "DOUBLE-CLICK ZOOM IN: panel=$clickedPanel (win=$(event.window_id))"
        zoomState.savedVerts = [copy(mainStates[i].calcDimsStruct.mainImageQuadVert) for i in panel_range]
        zoomState.savedVertSizes = [mainStates[i].calcDimsStruct.mainQuadVertSize for i in panel_range]
        zoomState.zoomedPanel = clickedPanel
        zoomState.isZoomed = true
        mainStates[1].switchIndex = clickedPanel

        zoomedCalcDim = getMainVerticies(mainStates[clickedPanel].calcDimsStruct, SingleImage, 1)
        mainStates[clickedPanel].calcDimsStruct = setproperties(
            mainStates[clickedPanel].calcDimsStruct,
            (mainImageQuadVert = zoomedCalcDim.mainImageQuadVert,
             mainQuadVertSize  = zoomedCalcDim.mainQuadVertSize))

        for i in panel_range
            if i != clickedPanel
                mainStates[i].calcDimsStruct = setproperties(
                    mainStates[i].calcDimsStruct,
                    (mainImageQuadVert = Float32[0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0],
                     mainQuadVertSize = 32 * sizeof(Float32)))
            end
        end
    else
        @info "DOUBLE-CLICK ZOOM OUT: restoring layout (win=$(event.window_id))"
        for (offset, i) in enumerate(panel_range)
            if offset <= length(zoomState.savedVerts)
                mainStates[i].calcDimsStruct = setproperties(
                    mainStates[i].calcDimsStruct,
                    (mainImageQuadVert = zoomState.savedVerts[offset],
                     mainQuadVertSize  = zoomState.savedVertSizes[offset]))
            end
        end
        zoomState.isZoomed = false
    end
end#reactToDoubleClick


"""
given list of cartesian coordinates and some window/ image characteristics - it translates mouse positions
to cartesian coordinates of the texture
strokeWidth - the property connected to the texture marking how thick should be the brush
mouseCoords - list of coordinates of mouse positions while left button remains pressed
calcDims - set of values usefull for calculating mouse position
return vector of translated cartesian coordinates
"""
function translateMouseToTexture(strokeWidth::Int32, mouseCoords::Vector{CartesianIndex{2}}, calcD::CalcDimsStruct, actualW::Int, actualH::Int)::Vector{CartesianIndex{2}}
    filteredList = Vector{CartesianIndex{2}}()
    for c in mouseCoords
        texX, texY = StructsManag.getTextureCoordinatesFromScreen(c[1], c[2], calcD, Float64(actualW), Float64(actualH))
        ix, iy = Int(round(texX)), Int(round(texY))
        if ix > 0 && iy > 0 && ix <= calcD.imageTextureWidth && iy <= calcD.imageTextureHeight
            push!(filteredList, CartesianIndex(ix, iy))
        end
    end

    if (!isempty(filteredList))
        return map(point -> addStrokeWidth(point, Int64(strokeWidth)), filteredList) |>  # adding some points around the point of choice so will be better visible
               (matrix) -> reduce(vcat, matrix) |># when we added some oints around we got list of lists so now we need to flatten it out
                           unique |> # we want only unique elements
                           uniq -> filter(it -> it[1] > 0 && it[1] <= calcD.imageTextureWidth && it[2] > 0 && it[2] <= calcD.imageTextureHeight, uniq)     # as we add new points they may end up getting outside the texture; we need to filter those out
    end #if
    #if we are here we do not have anything meaningfull else to return
    return Vector{CartesianIndex{2}}()
end #translateMouseToTexture

"""
adding the width to the stroke so we will be able to controll how thickly we are painting ...
"""
function addStrokeWidth(point::CartesianIndex{2}, strokeW::Int64)
    return CartesianIndices((-strokeW:strokeW, -strokeW:strokeW)) |> # set of cartesian indices that we will filter ot later
           list -> list .+ point |> # making coordinates around point of intrest
                   added -> filter(x -> (abs(point[1] - x[1]) + abs(x[2] - point[2])) < strokeW, added)# filtering to distant points
end#addStrokeWidth

# xx = CartesianIndex(2, 2)
# xx[1]
"""
helper function for translateMouseToTexture
"""
function getNewX(x::Int, calcD::CalcDimsStruct)::Int
    # first we subtract windowWidthCorr as in window the image do not need to start at the begining  of the window


    # 1) subtract from x the offset that is corrected width times widthCorr/2
    # 2) divide it by total width of the image taking into account offset from both sides
    # 3) now we have coordinate in range between 0 and 1 - relative coorde
    # 4) we get it to texture coordinates by multiplying by texture width
    # 5) we take min of texture width and result to clip it to max value
    # 6) we get max of 1 and result to avoid numbers less then 1

    return max(1, min(Int64(round(((x - (calcD.widthCorr * (calcD.corrected_width / 2))) / (calcD.corrected_width * (1 - calcD.widthCorr))) * calcD.imageTextureWidth)), calcD.imageTextureWidth))

    # In multi - image annotations for
    # LEFT IMAGE
    # return max(1,min(Int64(round(((x - calcD.widthCorr/2) / (calcD.corrected_width/2 + calcD.widthCorr)) * calcD.imageTextureWidth)), calcD.imageTextureWidth))
    # RIGHT IMAGE
    # return max(1, min(Int64(round(((x - (calcD.widthCorr / 2 + (calcD.corrected_width / 2))) / (calcD.corrected_width / 2 + calcD.widthCorr)) * calcD.imageTextureWidth)), calcD.imageTextureWidth))

end#getNewX

"""
helper function for translateMouseToTexture
"""
function getNewY(y::Int, calcD::CalcDimsStruct)::Int
    rounded_value = calcD.imageTextureHeight - round(((y - (calcD.heightCorr * (calcD.windowHeight / 2))) / (calcD.windowHeight * (1 - calcD.heightCorr))) * calcD.imageTextureHeight)

    clamped_value = clamp(rounded_value, 1, calcD.imageTextureHeight)
    return Int64(clamped_value)

end#getNewY

end #ReactOnMouseClickAndDrag



"""
left image

return max(1,min(Int64(round(((x - calcD.widthCorr/2) / (calcD.corrected_width/2 + calcD.widthCorr)) * calcD.imageTextureWidth)), calcD.imageTextureWidth))


right image
return max(1, min(Int64(round(((x - (calcD.widthCorr / 2 + (calcD.corrected_width / 2))) / (calcD.corrected_width / 2 + calcD.widthCorr)) * calcD.imageTextureWidth)), calcD.imageTextureWidth))


"""
