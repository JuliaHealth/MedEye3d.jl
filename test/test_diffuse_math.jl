using MedEye3d
using MedEye3d.VulkanBackend.VulkanContext
using MedEye3d.VulkanBackend.VulkanHeatDiffusion
using Vulkan
using GLFW

GLFW.Init()
GLFW.WindowHint(GLFW.CLIENT_API, GLFW.NO_API)
GLFW.WindowHint(GLFW.VISIBLE, 0)
window = GLFW.CreateWindow(256, 256, "Diff Math Test")
ctx = VulkanContext.init_vulkan_context(window, 256, 256)
state = HeatDiffusionState()
init_heat_diffusion!(state, ctx, 32, 32, 32)
diff = ones(Float32, 32, 32, 32)
VulkanHeatDiffusion.upload_diffusivity!(state, ctx, diff)
VulkanHeatDiffusion.seed_only!(state, ctx, 16, 16, 16; seed_radius=1)
VulkanHeatDiffusion.run_incremental_steps!(state, ctx, 1; copy_box_to_cpu=true)
mask = VulkanHeatDiffusion.read_mask_to_cpu(state, ctx)
println("Mask sum after 1 step: ", sum(mask))
destroy_heat_diffusion!(state, ctx)
GLFW.DestroyWindow(window)
GLFW.Terminate()

