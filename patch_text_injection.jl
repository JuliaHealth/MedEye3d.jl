path = "src/structs/Measurements.jl"
content = read(path, String)

sphere_text = """
                _push_thick_line!(vertices, u1, v1, u2, v2, color, w, h, 0.002f0)
            end
            
            # Draw SUV max next to the circle
            if m.suv_max != 0.0f0
                center_u = (cx - 0.5f0) / w
                center_v = (cy - 0.5f0) / h
                _draw_text_number!(vertices, m.suv_max, center_u + (r/w) + 0.01f0, center_v - 0.02f0, color, w, h)
            end
        end
"""
content = replace(content, """
                _push_thick_line!(vertices, u1, v1, u2, v2, color, w, h, 0.002f0)
            end
        end
""" => sphere_text)

line_text = """
            _push_thick_line!(vertices, u1, v1, u2, v2, color, w, h, 0.005f0)
            
            # Draw length next to the line
            if l.length_mm > 0.0f0
                mid_u = (u1 + u2) / 2.0f0
                mid_v = (v1 + v2) / 2.0f0
                _draw_text_number!(vertices, l.length_mm, mid_u + 0.01f0, mid_v - 0.02f0, color, w, h)
            end
        end
"""
content = replace(content, """
            _push_thick_line!(vertices, u1, v1, u2, v2, color, w, h, 0.005f0)
        end
""" => line_text)

write(path, content)
