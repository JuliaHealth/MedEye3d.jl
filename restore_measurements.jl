path = "src/display/reactingToMouseKeyboard/ReactOnMouseClickAndDrag.jl"
content = read(path, String)

funcs = """
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
        rx = ceil(Int, R / sx)
        ry = ceil(Int, R / sy)
        rz = ceil(Int, R / sz)
        cxi, cyi, czi = round(Int, cx_v), round(Int, cy_v), round(Int, cz_v)
        
        sum_suv = 0.0f0
        max_suv = 0.0f0
        count = 0
        
        for z in max(1, czi-rz):min(size(pet_dat, 3), czi+rz)
            dz = (z - czi) * sz
            for y in max(1, cyi-ry):min(size(pet_dat, 2), cyi+ry)
                dy = (y - cyi) * sy
                for x in max(1, cxi-rx):min(size(pet_dat, 1), cxi+rx)
                    dx = (x - cxi) * sx
                    dist = sqrt(dx^2 + dy^2 + dz^2)
                    if dist <= R
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
        
        # sample points along the line
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

function _handle_sphere_measurement(mousestr, mainStates, obj, center_idx, MEH, MeasMod)
    radius_mm = MEH.active_measurement_radius_mm[]
    
    # Find active measurement
    active_idx = findfirst(m -> m.is_active, obj.measurements)
    
    function trigger_obs(name)
        try
            LMW = MEH._get_lmw()
            if LMW !== nothing
                obs = getfield(LMW, :_lmw_observables)
                if haskey(obs, name)
                    obs[name][] += 1
                end
            end
        catch; end
    end
    
    # Only create/update sphere when button is held down (click-and-drag to place)
    if mousestr.isLeftButtonDown
        if active_idx === nothing
            # First click: create the sphere
            push!(obj.measurements, MeasMod.SphereMeasurement(
                id = length(obj.measurements) + 1,
                center_idx = center_idx,
                radius_mm = radius_mm,
                suv_mean = 0.0f0,
                suv_max = 0.0f0,
                is_active = true
            ))
            _compute_sphere_suv!(obj.measurements[end], mainStates)
            trigger_obs(:obs_refresh_measurements)
        else
            # Dragging: update position and compute SUV live!
            m = obj.measurements[active_idx]
            m.center_idx = center_idx
            m.radius_mm = radius_mm
            _compute_sphere_suv!(m, mainStates)
            trigger_obs(:obs_update_measurements)
        end
    else
        # Mouse released: finalize
        if active_idx !== nothing
            m = obj.measurements[active_idx]
            m.center_idx = center_idx
            m.radius_mm = radius_mm
            m.is_active = false # Saved!
            _compute_sphere_suv!(m, mainStates)
            trigger_obs(:obs_update_measurements)
            trigger_obs(:obs_refresh_measurements)
        end
    end
end

function _handle_line_measurement(mousestr, mainStates, obj, point3d, MEH, MeasMod)
    # Find active line (start placed, waiting for end)
    active_idx = findfirst(m -> m.is_active, obj.line_measurements)
    
    # Helper to trigger observables
    function trigger_obs(name)
        try
            LMW = MEH._get_lmw()
            if LMW !== nothing
                obs = getfield(LMW, :_lmw_observables)
                if haskey(obs, name)
                    obs[name][] += 1
                end
            end
        catch; end
    end
    
    if mousestr.isLeftButtonDown
        if active_idx === nothing
            # Start of click-and-drag: create new line
            push!(obj.line_measurements, MeasMod.LineMeasurement(
                id = length(obj.line_measurements) + 1,
                start_idx = point3d,
                end_idx = point3d,
                length_mm = 0.0f0,
                suv_mean = 0.0f0,
                suv_max = 0.0f0,
                is_active = true
            ))
            _compute_line_suv!(obj.line_measurements[end], mainStates)
            trigger_obs(:obs_refresh_measurements)
        else
            # Dragging: update endpoint
            lm = obj.line_measurements[active_idx]
            lm.end_idx = point3d
            
            axialState = mainStates[1]
            sp = axialState.spacingsValue[1]
            sx, sy, sz = Float32(sp[1]), Float32(sp[2]), Float32(sp[3])
            dx_mm = (lm.end_idx[1] - lm.start_idx[1]) * sx
            dy_mm = (lm.end_idx[2] - lm.start_idx[2]) * sy
            dz_mm = (lm.end_idx[3] - lm.start_idx[3]) * sz
            lm.length_mm = sqrt(dx_mm^2 + dy_mm^2 + dz_mm^2)
            _compute_line_suv!(lm, mainStates)
            
            trigger_obs(:obs_update_measurements)
        end
    else
        # Mouse released: finalize the line if active
        if active_idx !== nothing
            lm = obj.line_measurements[active_idx]
            lm.end_idx = point3d
            lm.is_active = false
            
            axialState = mainStates[1]
            sp = axialState.spacingsValue[1]
            sx, sy, sz = Float32(sp[1]), Float32(sp[2]), Float32(sp[3])
            dx_mm = (lm.end_idx[1] - lm.start_idx[1]) * sx
            dy_mm = (lm.end_idx[2] - lm.start_idx[2]) * sy
            dz_mm = (lm.end_idx[3] - lm.start_idx[3]) * sz
            lm.length_mm = sqrt(dx_mm^2 + dy_mm^2 + dz_mm^2)
            _compute_line_suv!(lm, mainStates)
            
            trigger_obs(:obs_update_measurements)
            trigger_obs(:obs_refresh_measurements) # to update color/state
        end
    end
end
"""

content = replace(content, "function reactToMouseDrag(" => funcs * "\nfunction reactToMouseDrag(")

call_injection = """
        if panelState.moveLesionMode
            # Find active lesion and update position
            # ... already in the code ...
"""

# We need to inject the call to `_handle_sphere_measurement` right after coordinate mapping.
call_to_inject = """
    # Determine measurement sub-mode
    MEH = parentmodule(@__MODULE__).SegmentationDisplay.MakieEventHandlers
    sub_mode = MEH.measurement_sub_mode[]
    MeasMod = parentmodule(@__MODULE__).Measurements
    
    if sub_mode == :sphere
        _handle_sphere_measurement(mousestr, mainStates, panelState.mainForDisplayObjects, (Float32(origX), Float32(origY), Float32(origZ)), MEH, MeasMod)
    elseif sub_mode == :line
        _handle_line_measurement(mousestr, mainStates, panelState.mainForDisplayObjects, (Float32(origX), Float32(origY), Float32(origZ)), MEH, MeasMod)
    end
    
    if panelState.moveLesionMode
"""

content = replace(content, "    if panelState.moveLesionMode\n" => call_to_inject)

write(path, content)
println("Restored measurement functionality properly!")
