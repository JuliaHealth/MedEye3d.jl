"""
PrepareWindowHelpers stub — OpenGL buffer creation removed, Vulkan backend handles buffers.
"""
module PrepareWindowHelpers
using GLFW, Logging

export initializeWindow, createVertexBuffer, createDAtaBuffer, createDynamicDAtaBuffer
export createElementBuffer, createDynamicElementBuffer, encodeDataFromDataBuffer, controllWindowInput

function initializeWindow(width::Int, height::Int)
    # Create GLFW window with NO_API for Vulkan
    GLFW.WindowHint(GLFW.CLIENT_API, GLFW.NO_API)
    GLFW.WindowHint(GLFW.RESIZABLE, true)
    window = GLFW.CreateWindow(width, height, "MedEye3d")
    # Reset CLIENT_API hint so GLMakie (Makie control window) can create OpenGL windows later
    GLFW.WindowHint(GLFW.CLIENT_API, GLFW.OPENGL_API)
    return window
end

# Stubs for OpenGL buffer functions — not used by Vulkan backend
function createVertexBuffer(); return Ref(UInt32(0)); end
function createDAtaBuffer(vertices); return Ref(UInt32(0)); end
function createDynamicDAtaBuffer(vertices); return Ref(UInt32(0)); end
function createElementBuffer(indices); return Ref(UInt32(0)); end
function createDynamicElementBuffer(indices); return Ref(UInt32(0)); end
function encodeDataFromDataBuffer(); end
function controllWindowInput(window); end

end # module PrepareWindowHelpers
