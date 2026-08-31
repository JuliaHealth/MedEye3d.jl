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
    pollingTask = @task begin
        try
            while true
                if isready(stopChannel)
                    take!(stopChannel)
                    break
                end
                if GLFW.WindowShouldClose(window)
                    break
                end
                # Process all pending GLFW events (mouse, keyboard, scroll, resize).
                # Without this call, GLFW callbacks are never invoked and the window
                # appears frozen. Previously this relied on the Makie renderloop, but
                # that created a fragile dependency that broke when GLMakie failed.
                try
                    GLFW.PollEvents()
                catch e
                    # PollEvents can throw if window was destroyed concurrently
                    if e isa GLFW.GLFWError && e.code == GLFW.NOT_INITIALIZED
                        break
                    end
                end
                sleep(0.008)
            end
        catch e
            @warn "GLFW polling task error: \$e" exception=(e, catch_backtrace())
        finally
            @info "GLFW polling task ended"
        end
    end
    schedule(pollingTask)

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
