path = "src/display/reactingToMouseKeyboard/ReactOnMouseClickAndDrag.jl"
content = read(path, String)

insert_code = """
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
"""

# Inject functions
content = replace(content, "function _handle_sphere_measurement" => insert_code * "\nfunction _handle_sphere_measurement")

# Fix inline sphere calculation
inline_sphere_calc = r"            # Compute SUV live while dragging\n            axialState = mainStates\[1\].*?                m\.suv_max = 0\.0f0\n                end\n            end"s
content = replace(content, inline_sphere_calc => "            _compute_sphere_suv!(m, mainStates)")

# Fix sphere creation (in _handle_sphere_measurement)
s_match = """                suv_max = 0.0f0,
                is_active = true
            ))
            trigger_obs(:obs_refresh_measurements)"""
s_repl = """                suv_max = 0.0f0,
                is_active = true
            ))
            _compute_sphere_suv!(obj.measurements[end], mainStates)
            trigger_obs(:obs_refresh_measurements)"""
content = replace(content, s_match => s_repl)

# Fix line creation (in _handle_line_measurement)
l_match = """                length_mm = 0.0f0,
                is_active = true
            ))
            trigger_obs(:obs_refresh_measurements)"""
l_repl = """                length_mm = 0.0f0,
                is_active = true
            ))
            _compute_line_suv!(obj.line_measurements[end], mainStates)
            trigger_obs(:obs_refresh_measurements)"""
content = replace(content, l_match => l_repl)

# Fix line dragging
l_drag_match = r"            axialState = mainStates\[1\]\n            sp = axialState\.spacingsValue\[1\]\n            sx, sy, sz = Float32\(sp\[1\]\), Float32\(sp\[2\]\), Float32\(sp\[3\]\)\n            dx_mm = \(lm\.end_idx\[1\] - lm\.start_idx\[1\]\) \* sx\n            dy_mm = \(lm\.end_idx\[2\] - lm\.start_idx\[2\]\) \* sy\n            dz_mm = \(lm\.end_idx\[3\] - lm\.start_idx\[3\]\) \* sz\n            lm\.length_mm = sqrt\(dx_mm\^2 \+ dy_mm\^2 \+ dz_mm\^2\)"
l_drag_repl = """            axialState = mainStates[1]
            sp = axialState.spacingsValue[1]
            sx, sy, sz = Float32(sp[1]), Float32(sp[2]), Float32(sp[3])
            dx_mm = (lm.end_idx[1] - lm.start_idx[1]) * sx
            dy_mm = (lm.end_idx[2] - lm.start_idx[2]) * sy
            dz_mm = (lm.end_idx[3] - lm.start_idx[3]) * sz
            lm.length_mm = sqrt(dx_mm^2 + dy_mm^2 + dz_mm^2)
            _compute_line_suv!(lm, mainStates)"""
content = replace(content, l_drag_match => l_drag_repl)

write(path, content)
println("Patched SUV logic properly")
