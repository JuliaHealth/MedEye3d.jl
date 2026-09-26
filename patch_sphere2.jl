path = "src/display/reactingToMouseKeyboard/ReactOnMouseClickAndDrag.jl"
content = read(path, String)

sphere_logic = """
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
            trigger_obs(:obs_refresh_measurements)
        else
            # Dragging: update position and compute SUV live!
            m = obj.measurements[active_idx]
            m.center_idx = center_idx
            m.radius_mm = radius_mm
            
            # Compute SUV live
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
                if sz == 0.0f0; sz = 1.0f0; end
                
                cxi, cyi, czi = round(Int, m.center_idx[1]), round(Int, m.center_idx[2]), round(Int, m.center_idx[3])
                R = m.radius_mm
                rx = ceil(Int, R / sx)
                ry = ceil(Int, R / sy)
                rz = ceil(Int, R / sz)
                
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
                end
            end
            trigger_obs(:obs_update_measurements)
        end
    else
        # Mouse released: finalize
        if active_idx !== nothing
            m = obj.measurements[active_idx]
            m.center_idx = center_idx
            m.radius_mm = radius_mm
            m.is_active = false # Saved!
            trigger_obs(:obs_update_measurements)
            trigger_obs(:obs_refresh_measurements)
        end
    end
end
"""

content = replace(content, r"function _handle_sphere_measurement.*?end\nend\nend\nend"s => sphere_logic)
write(path, content)
