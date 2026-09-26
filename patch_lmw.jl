path = "src/display/LesionMetadataWindow.jl"
content = read(path, String)

old_code = """
        obj = stateObjects[1].mainForDisplayObjects
        saved_spheres = filter(m -> !m.is_active, obj.measurements)
        saved_lines = filter(m -> !m.is_active, obj.line_measurements)
        
        if isempty(saved_spheres) && isempty(saved_lines)
            Label(grid[1, 1:4], "No measurements yet. Enable mode, then click.",
                fontsize=9, color=RGBf(0.5, 0.5, 0.5), halign=:center)
            return
        end
        
        row_i = 0
        
        # Sphere measurements
        if !isempty(saved_spheres)
            row_i += 1
            Label(grid[row_i, 1:4], "-- SUV Spheres --",
                fontsize=9, color=RGBf(1.0, 1.0, 0.3), halign=:center)
        end
        for (i, m) in enumerate(saved_spheres)
            row_i += 1
            Label(grid[row_i, 1], "S#\$(m.id) R=\$(round(m.radius_mm, digits=1))mm",
                fontsize=9, color=RGBf(1.0, 1.0, 0.3), halign=:left)
            Label(grid[row_i, 2], "SUV: \$(round(m.suv_mean, digits=2)) / \$(round(m.suv_max, digits=2))",
                fontsize=9, color=TXT, halign=:left)
            
            btn_eye = Button(grid[row_i, 3], label="[>]", buttoncolor=BG_PNL, labelcolor=TXT, fontsize=9)
            btn_del = Button(grid[row_i, 4], label="[x]", buttoncolor=BG_PNL, labelcolor=RGBf(0.9, 0.3, 0.3), fontsize=9)
            
            local mid = m.id
            on(btn_eye.clicks) do _
                try
                    channel.put(MakieEvents.JumpToMeasurementEvent(mid))
                catch; end
            end
            on(btn_del.clicks) do _
                try
                    channel.put(MakieEvents.DeleteMeasurementEvent(mid))
                catch; end
            end
        end
        
        # Line measurements
        if !isempty(saved_lines)
            row_i += 1
            Label(grid[row_i, 1:4], "-- Line Distances --",
                fontsize=9, color=RGBf(0.3, 1.0, 1.0), halign=:center)
        end
        for (i, lm) in enumerate(saved_lines)
            row_i += 1
            len_str = lm.length_mm >= 10.0f0 ? "\$(round(lm.length_mm / 10.0f0, digits=2))cm" : "\$(round(lm.length_mm, digits=1))mm"
            Label(grid[row_i, 1:2], "L#\$(lm.id): \$len_str",
                fontsize=9, color=RGBf(0.3, 1.0, 1.0), halign=:left)
            
            btn_eye_l = Button(grid[row_i, 3], label="[>]", buttoncolor=BG_PNL, labelcolor=TXT, fontsize=9)
            btn_del_l = Button(grid[row_i, 4], label="[x]", buttoncolor=BG_PNL, labelcolor=RGBf(0.9, 0.3, 0.3), fontsize=9)
            
            local lid = lm.id
            on(btn_eye_l.clicks) do _
                try
                    channel.put(MakieEvents.JumpToLineMeasurementEvent(lid))
                catch; end
            end
            on(btn_del_l.clicks) do _
                try
                    channel.put(MakieEvents.DeleteLineMeasurementEvent(lid))
                catch; end
            end
        end
"""

new_code = """
        if !haskey(_lmw_observables, :obs_update_measurements)
            _lmw_observables[:obs_update_measurements] = Observable(0)
        end
        obs_upd = _lmw_observables[:obs_update_measurements]

        obj = stateObjects[1].mainForDisplayObjects
        # DO NOT filter out active ones, show them all!
        saved_spheres = obj.measurements
        saved_lines = obj.line_measurements
        
        if isempty(saved_spheres) && isempty(saved_lines)
            Label(grid[1, 1:4], "No measurements yet. Enable mode, then click.",
                fontsize=9, color=RGBf(0.5, 0.5, 0.5), halign=:center)
            return
        end
        
        row_i = 0
        
        # Sphere measurements
        if !isempty(saved_spheres)
            row_i += 1
            Label(grid[row_i, 1:4], "-- SUV Spheres --",
                fontsize=9, color=RGBf(1.0, 1.0, 0.3), halign=:center)
        end
        for (i, m) in enumerate(saved_spheres)
            row_i += 1
            
            local m_ref = m
            lbl1 = lift(obs_upd) do _
                "S#\$(m_ref.id) R=\$(round(m_ref.radius_mm, digits=1))mm"
            end
            lbl2 = lift(obs_upd) do _
                "SUV: \$(round(m_ref.suv_mean, digits=2)) / \$(round(m_ref.suv_max, digits=2))"
            end
            color_obs = lift(obs_upd) do _
                m_ref.is_active ? RGBf(1.0, 0.4, 0.4) : RGBf(1.0, 1.0, 0.3)
            end
            
            Label(grid[row_i, 1], lbl1,
                fontsize=9, color=color_obs, halign=:left)
            Label(grid[row_i, 2], lbl2,
                fontsize=9, color=TXT, halign=:left)
            
            btn_eye = Button(grid[row_i, 3], label="[>]", buttoncolor=BG_PNL, labelcolor=TXT, fontsize=9)
            btn_del = Button(grid[row_i, 4], label="[x]", buttoncolor=BG_PNL, labelcolor=RGBf(0.9, 0.3, 0.3), fontsize=9)
            
            local mid = m.id
            on(btn_eye.clicks) do _
                try
                    channel.put(MakieEvents.JumpToMeasurementEvent(mid))
                catch; end
            end
            on(btn_del.clicks) do _
                try
                    channel.put(MakieEvents.DeleteMeasurementEvent(mid))
                catch; end
            end
        end
        
        # Line measurements
        if !isempty(saved_lines)
            row_i += 1
            Label(grid[row_i, 1:4], "-- Line Distances --",
                fontsize=9, color=RGBf(0.3, 1.0, 1.0), halign=:center)
        end
        for (i, lm) in enumerate(saved_lines)
            row_i += 1
            
            local lm_ref = lm
            lbl1 = lift(obs_upd) do _
                len_str = lm_ref.length_mm >= 10.0f0 ? "\$(round(lm_ref.length_mm / 10.0f0, digits=2))cm" : "\$(round(lm_ref.length_mm, digits=1))mm"
                "L#\$(lm_ref.id): \$len_str"
            end
            color_obs = lift(obs_upd) do _
                lm_ref.is_active ? RGBf(1.0, 0.4, 0.4) : RGBf(0.3, 1.0, 1.0)
            end
            
            Label(grid[row_i, 1:2], lbl1,
                fontsize=9, color=color_obs, halign=:left)
            
            btn_eye_l = Button(grid[row_i, 3], label="[>]", buttoncolor=BG_PNL, labelcolor=TXT, fontsize=9)
            btn_del_l = Button(grid[row_i, 4], label="[x]", buttoncolor=BG_PNL, labelcolor=RGBf(0.9, 0.3, 0.3), fontsize=9)
            
            local lid = lm.id
            on(btn_eye_l.clicks) do _
                try
                    channel.put(MakieEvents.JumpToLineMeasurementEvent(lid))
                catch; end
            end
            on(btn_del_l.clicks) do _
                try
                    channel.put(MakieEvents.DeleteLineMeasurementEvent(lid))
                catch; end
            end
        end
"""

content = replace(content, old_code => new_code)
write(path, content)
