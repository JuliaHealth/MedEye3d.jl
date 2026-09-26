path = "src/display/reactingToMouseKeyboard/ReactOnMouseClickAndDrag.jl"
content = read(path, String)

line_logic = """
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
                is_active = true
            ))
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
            
            trigger_obs(:obs_update_measurements)
            trigger_obs(:obs_refresh_measurements) # to update color/state
        end
    end
end
"""
content = replace(content, r"function _handle_line_measurement.*?end\nend"s => line_logic)
write(path, content)
