"""
ShadersAndVerticies stub — OpenGL shaders removed, Vulkan backend uses VulkanShaders.
"""
module ShadersAndVerticies


export createVertexShader, createFragmentShader

# Vertex data kept for CalcDimsStruct geometry calculations
const elements = UInt32[0, 1, 2, 2, 3, 0]

function createVertexShader(gslsStr); return UInt32(0); end
function createFragmentShader(gslsStr, textSpecs, color); return UInt32(0); end

end # module ShadersAndVerticies
