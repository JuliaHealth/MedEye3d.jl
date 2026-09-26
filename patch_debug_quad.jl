path = "src/structs/Measurements.jl"
content = read(path, String)
content = replace(content, """
    return vertices
end

function compute_line_vertices(line_measurements::Vector{LineMeasurement}, state, panel_id::Int)::Vector{Float32}
""" => """
    if panel_id == 1
        append!(vertices, [
            0.25f0, 0.25f0, 0.0f0, 1.0f0, 0.0f0, 1.0f0,
            0.75f0, 0.25f0, 0.0f0, 1.0f0, 0.0f0, 1.0f0,
            0.25f0, 0.75f0, 0.0f0, 1.0f0, 0.0f0, 1.0f0,
            0.25f0, 0.75f0, 0.0f0, 1.0f0, 0.0f0, 1.0f0,
            0.75f0, 0.25f0, 0.0f0, 1.0f0, 0.0f0, 1.0f0,
            0.75f0, 0.75f0, 0.0f0, 1.0f0, 0.0f0, 1.0f0
        ])
    end
    return vertices
end

function compute_line_vertices(line_measurements::Vector{LineMeasurement}, state, panel_id::Int)::Vector{Float32}
""")
write(path, content)
