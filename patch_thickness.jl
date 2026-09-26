path = "src/structs/Measurements.jl"
content = read(path, String)

content = replace(content, 
    "function _push_thick_line!(vertices::Vector{Float32}, u1, v1, u2, v2, color, w, h)" => 
    "function _push_thick_line!(vertices::Vector{Float32}, u1, v1, u2, v2, color, w, h, thickness::Float32=0.005f0)"
)
content = replace(content, "t = 0.040f0" => "t = thickness")

content = replace(content, 
    "_push_thick_line!(vertices, u1, v1, u2, v2, color, w, h)" => 
    "_push_thick_line!(vertices, u1, v1, u2, v2, color, w, h, 0.003f0)"
)

# Wait, is _push_thick_line! called for lines? Let's check `compute_line_vertices`.
