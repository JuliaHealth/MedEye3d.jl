path = "src/display/reactingToMouseKeyboard/ReactOnMouseClickAndDrag.jl"
content = read(path, String)

sphere_patch = """
        if active_idx !== nothing
            s = obj.measurements[active_idx]
            obj.measurements[active_idx] = MeasMod.SphereMeasurement(
                id = s.id, center_idx = s.center_idx, radius_mm = s.radius_mm,
                suv_mean = s.suv_mean, suv_max = s.suv_max,
                is_active = false
            )
            
            # Notify LMW to refresh measurement list
            try
                LMW = MEH._get_lmw()
                if LMW !== nothing
                    obs = getfield(LMW, :_lmw_observables)
                    if haskey(obs, :obs_refresh_measurements)
                        obs[:obs_refresh_measurements][] += 1
                    end
                end
            catch; end
        end
"""
content = replace(content, """
        if active_idx !== nothing
            s = obj.measurements[active_idx]
            obj.measurements[active_idx] = MeasMod.SphereMeasurement(
                id = s.id, center_idx = s.center_idx, radius_mm = s.radius_mm,
                suv_mean = s.suv_mean, suv_max = s.suv_max,
                is_active = false
            )
        end
""" => sphere_patch)

line_patch = """
            dz_mm = (lm.end_idx[3] - lm.start_idx[3]) * sz
            lm.length_mm = sqrt(dx_mm^2 + dy_mm^2 + dz_mm^2)
            
            # Notify LMW to refresh measurement list
            try
                LMW = MEH._get_lmw()
                if LMW !== nothing
                    obs = getfield(LMW, :_lmw_observables)
                    if haskey(obs, :obs_refresh_measurements)
                        obs[:obs_refresh_measurements][] += 1
                    end
                end
            catch; end
        end
"""
content = replace(content, """
            dz_mm = (lm.end_idx[3] - lm.start_idx[3]) * sz
            lm.length_mm = sqrt(dx_mm^2 + dy_mm^2 + dz_mm^2)
        end
""" => line_patch)

write(path, content)
