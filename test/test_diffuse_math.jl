using MedEye3d
using MedEye3d.VulkanBackend.VulkanContext
using MedEye3d.VulkanBackend.VulkanHeatDiffusion
using Vulkan

ctx = VulkanContext.init_vulkan_context()
state = VulkanHeatDiffusion.init_heat_diffusion_gpu(ctx, 32, 32, 32)
diff = ones(Float32, 32, 32, 32)
VulkanHeatDiffusion.upload_diffusivity!(state, ctx, diff)
VulkanHeatDiffusion.seed_only!(state, ctx, 16, 16, 16; seed_radius=1)
VulkanHeatDiffusion.run_incremental_steps!(state, ctx, 1; copy_box_to_cpu=true)
mask = VulkanHeatDiffusion.read_mask_to_cpu(state, ctx)
println("Mask sum after 1 step: ", sum(mask))
