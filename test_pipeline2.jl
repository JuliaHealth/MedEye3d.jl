using Pkg; Pkg.activate(".")
import GLFW
GLFW.Init()
GLFW.WindowHint(GLFW.CLIENT_API, GLFW.NO_API)
GLFW.WindowHint(GLFW.VISIBLE, 0)
win = GLFW.CreateWindow(800, 600, "Hidden")

include("src/display/Vulkan/VulkanBackend.jl")
using .VulkanBackend.VulkanContext
using Vulkan

ctx = VulkanContext.init_vulkan_context(win, 800, 600)
dev = ctx.device

res = Vulkan.create_compute_pipelines(dev, [])
println(typeof(res))

unwrap = VulkanContext.unwrap
println(typeof(unwrap(res)))

GLFW.DestroyWindow(win)
GLFW.Terminate()
