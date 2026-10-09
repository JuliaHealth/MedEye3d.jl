using MedEye3d
using MedEye3d.VulkanContext
using MedEye3d.VulkanHeatDiffusion

ctx = init_vulkan_context()
state = VulkanHeatDiffusion.HeatDiffusionState()
w, h, d = 512, 512, 359
println("Init heat diffusion...")
VulkanHeatDiffusion.init_heat_diffusion!(state, ctx, w, h, d)

println("Allocating diffusivity...")
diff_vol = ones(Float32, w, h, d)

println("Uploading diffusivity...")
VulkanHeatDiffusion.upload_diffusivity!(state, ctx, diff_vol)
println("Success!")
