path = "src/display/reactingToMouseKeyboard/ReactOnMouseClickAndDrag.jl"
content = read(path, String)

# We will replace the entire _handle_sphere_measurement function
new_func = """
function _handle_sphere_measurement(mousestr, mainStates, obj, center_idx, MEH, MeasMod)
    radius_mm = MEH.active_measurement_radius_mm[]
    
    # Find active measurement (currently being drawn)
    active_idx = findfirst(m -> m.is_active, obj.measurements)
    
    # Helper to trigger observables
    function trigger_obs(name)
        try
            LMW = MEH._get_lmw()
            if LMW !== nothing
                obs = getfield(LMW, :_lmw_observables)
                if haskey(obs, name)
                    if typeof(obs[name][]) == Int
                        obs[name][] += 1
                    else
                        obs[name][] = obj
                    end
                end
            end
        catch; end
    end
    
    if mousestr.isLeftButtonDown
        if active_idx === nothing
            # First click: Create a new active sphere
            push!(obj.measurements, MeasMod.SphereMeasurement(
                id = length(obj.measurements) + 1,
                center_idx = center_idx,
                radius_mm = radius_mm,
                suv_mean = 0.0f0,
                suv_max = 0.0f0,
                is_active = true
            ))
            trigger_obs(:obs_refresh_measurements)
        else
            # Dragging: Update active sphere position and radius
            m = obj.measurements[active_idx]
            m.center_idx = center_idx
            m.radius_mm = radius_mm
            
            # Compute SUV live while dragging
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
            
            # Trigger fast UI update (no list rebuild)
            trigger_obs(:obs_update_measurements)
        end
    else
        # Mouse released
        if active_idx !== nothing
            # Lock it in
            m = obj.measurements[active_idx]
            m.is_active = false
            trigger_obs(:obs_update_measurements)
            trigger_obs(:obs_refresh_measurements) # to update color/state
        end
    end
end
"""

# Find the old function
start_idx = findfirst("function _handle_sphere_measurement", content)
# We know the next function is _handle_line_measurement
end_idx = findfirst("function _handle_line_measurement", content)

if start_idx !== nothing && end_idx !== nothing
    content = content[1:start_idx[1]-1] * new_func * content[end_idx[1]:end]
    write(path, content)
    println("Patched _handle_sphere_measurement")
else
    println("Failed to find boundaries")
end
