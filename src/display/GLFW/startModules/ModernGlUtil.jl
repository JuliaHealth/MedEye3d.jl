"""
ModernGlUtil stub — OpenGL backend removed, kept for module inclusion compatibility.
"""
module ModernGlUtil
using GLFW, Logging

export createcontextinfo, get_glsl_version_string

function createcontextinfo()
    # No-op: Vulkan backend doesn't need OpenGL context info
end

function get_glsl_version_string()
    return ""  # Not used by Vulkan backend
end

end # module ModernGlUtil
