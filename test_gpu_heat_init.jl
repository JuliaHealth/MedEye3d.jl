using Pkg; Pkg.activate(".")
import GLFW
GLFW.Init()
GLFW.WindowHint(GLFW.CLIENT_API, GLFW.NO_API)
GLFW.WindowHint(GLFW.VISIBLE, 0)
win = GLFW.CreateWindow(800, 600, "Hidden")

include("src/display/Vulkan/VulkanBackend.jl")
using .VulkanBackend.VulkanContext
using .VulkanBackend.VulkanHeatDiffusion

ctx = VulkanContext.init_vulkan_context(win, 800, 600)
println("Vulkan context initialized successfully")

w, h, d = 512, 512, 359
heat_state = VulkanHeatDiffusion.HeatDiffusionState()
try
    VulkanHeatDiffusion.init_heat_diffusion!(heat_state, ctx, w, h, d)
    println("HeatDiffusion GPU initialized successfully!")
catch e
    println("ERROR during init_heat_diffusion!:")
    showerror(stdout, e, catch_backtrace())
end

GLFW.DestroyWindow(win)
GLFW.Terminate()
