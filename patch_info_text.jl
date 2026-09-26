path = "src/display/reactingToMouseKeyboard/ReactOnMouseClickAndDrag.jl"
content = read(path, String)

old_text = "            info_str = join(parts, \" | \")\n            MEH.cursor_info_text[] = info_str"

new_text = """
            # Add visible measurements to info text
            meas_parts = String[]
            spacing_z = panelState.calcDimsStruct.voxelSizeZ
            if spacing_z == 0; spacing_z = 1.0; end
            
            for m in panelState.mainForDisplayObjects.measurements
                dz = abs(m.center_idx[3] - currentSlice) * spacing_z
                if dz <= m.radius_mm || m.is_active
                    push!(meas_parts, "Sphere(" * string(round(m.radius_mm, digits=1)) * "mm) SUVmax: " * string(round(m.suv_max, digits=2)))
                end
            end
            for m in panelState.mainForDisplayObjects.line_measurements
                if abs(m.start_idx[3] - currentSlice) < 3 || abs(m.end_idx[3] - currentSlice) < 3 || m.is_active
                    push!(meas_parts, "Line(" * string(round(m.length_mm, digits=1)) * "mm) SUVmax: " * string(round(m.suv_max, digits=2)))
                end
            end
            
            if !isempty(meas_parts)
                push!(parts, "MEASUREMENTS: " * join(meas_parts, ", "))
            end

            info_str = join(parts, " | ")
            MEH.cursor_info_text[] = info_str
"""

content = replace(content, old_text => new_text)
write(path, content)
