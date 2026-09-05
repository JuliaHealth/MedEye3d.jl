"""
ShadersAndVerticiesForLine stub — OpenGL line shaders removed, crosshair rendering handled by Vulkan overlay.
"""
module ShadersAndVerticiesForLine

export createAndInitLineShaderProgram, line_vertices, line_indices

const line_vertices = Float32[]
const line_indices = UInt32[]

function createAndInitLineShaderProgram(vertex_shader)
    return (UInt32(0), UInt32(0))
end

end # module ShadersAndVerticiesForLine
