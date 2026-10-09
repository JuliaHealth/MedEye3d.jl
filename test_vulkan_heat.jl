using Pkg; Pkg.activate(".")
include("src/display/Vulkan/VulkanBackend.jl")
using .VulkanBackend.VulkanContext
using .VulkanBackend.VulkanHeatDiffusion

ctx = VulkanContext.init_vulkan_context()
println("Context initialized")
heat_state = VulkanHeatDiffusion.HeatDiffusionState()
w, h, d = 512, 512, 359
println("Init heat diffusion...")
try
    VulkanHeatDiffusion.init_heat_diffusion!(heat_state, ctx, w, h, d)
    println("SUCCESS")
catch e
    println("FAILED:")
    showerror(stdout, e, catch_backtrace())
end
