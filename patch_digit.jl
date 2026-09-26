path = "src/structs/Measurements.jl"
content = read(path, String)

digit_code = """
const DIGIT_SEGMENTS = [
    [0, 1, 2, 4, 5, 6],       # 0
    [2, 5],                   # 1
    [0, 2, 3, 4, 6],          # 2
    [0, 2, 3, 5, 6],          # 3
    [1, 2, 3, 5],             # 4
    [0, 1, 3, 5, 6],          # 5
    [0, 1, 3, 4, 5, 6],       # 6
    [0, 2, 5],                # 7
    [0, 1, 2, 3, 4, 5, 6],    # 8
    [0, 1, 2, 3, 5, 6]        # 9
]

function _draw_digit!(vertices, digit::Int, u, v, dw, dh, color, w, h, thickness)
    if digit < 0 || digit > 9; return; end
    segs = DIGIT_SEGMENTS[digit + 1]
    
    # 0: top
    if 0 in segs; _push_thick_line!(vertices, u, v, u+dw, v, color, w, h, thickness); end
    # 1: top-left
    if 1 in segs; _push_thick_line!(vertices, u, v, u, v+dh/2, color, w, h, thickness); end
    # 2: top-right
    if 2 in segs; _push_thick_line!(vertices, u+dw, v, u+dw, v+dh/2, color, w, h, thickness); end
    # 3: middle
    if 3 in segs; _push_thick_line!(vertices, u, v+dh/2, u+dw, v+dh/2, color, w, h, thickness); end
    # 4: bottom-left
    if 4 in segs; _push_thick_line!(vertices, u, v+dh/2, u, v+dh, color, w, h, thickness); end
    # 5: bottom-right
    if 5 in segs; _push_thick_line!(vertices, u+dw, v+dh/2, u+dw, v+dh, color, w, h, thickness); end
    # 6: bottom
    if 6 in segs; _push_thick_line!(vertices, u, v+dh, u+dw, v+dh, color, w, h, thickness); end
end

function _draw_text_number!(vertices, val::Float32, u, v, color, w, h)
    # Convert to string with 1 decimal place
    str = string(round(val, digits=1))
    
    dw = 0.015f0
    dh = 0.030f0
    spacing = 0.020f0
    thickness = 0.003f0
    
    curr_u = u
    for char in str
        if char == '.'
            # Draw a small dot
            _push_thick_line!(vertices, curr_u, v+dh, curr_u+0.002f0, v+dh, color, w, h, thickness)
            curr_u += 0.010f0
        elseif char == '-'
            # Draw a minus
            _push_thick_line!(vertices, curr_u, v+dh/2, curr_u+dw, v+dh/2, color, w, h, thickness)
            curr_u += spacing
        else
            digit = parse(Int, string(char))
            _draw_digit!(vertices, digit, curr_u, v, dw, dh, color, w, h, thickness)
            curr_u += spacing
        end
    end
end
"""

# Insert before compute_measurement_vertices
content = replace(content, "function compute_measurement_vertices" => digit_code * "\n\nfunction compute_measurement_vertices")
write(path, content)
