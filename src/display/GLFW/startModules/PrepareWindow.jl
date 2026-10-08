"""
PrepareWindow stub — OpenGL window/shader preparation removed.
Vulkan backend creates GLFW window with NO_API and handles initialization separately.
"""
module PrepareWindow
using Base.Threads, GLFW, Logging
using ..PrepareWindowHelpers, ..DataStructs, ..ForDisplayStructs

export displayAll, createAndInitShaderProgram

"""
Create GLFW window with NO_API for Vulkan backend, set up polling task.
"""
function displayAll(calcDimsStruct::CalcDimsStruct)
    window = PrepareWindowHelpers.initializeWindow(calcDimsStruct.windowWidth, calcDimsStruct.windowHeight)
    stopChannel = Channel{Bool}(1)
    # Return compatible tuple (stubs for shader/buffer handles)
    return (window, UInt32(0), Ref(UInt32(0)), Ref(UInt32(0)), UInt32(0), Ref(UInt32(0)), UInt32(0), "", stopChannel)
end

"""
Create shader program stub — Vulkan shaders are created by VulkanShaders module.
"""
function createAndInitShaderProgram(vertex_shader::UInt32, listOfTexturesToCreate::Vector{TextureSpec{Float32}}, gslsStr::String, calcDimsStruct::CalcDimsStruct, num)
    return (UInt32(0), UInt32(0), Ref(UInt32(0)))
end

end # module PrepareWindow
