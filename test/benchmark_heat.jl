using MedEye3d
using MedEye3d.VulkanBackend.VulkanHeatDiffusion
using MedEye3d.VulkanBackend.VulkanContext
using Vulkan

println("Initializing Vulkan context...")
ctx = VulkanContext.init_vulkan_context()
w, h, d = 64, 64, 64
println("Initializing HeatDiffusionState...")
state = VulkanHeatDiffusion.init_heat_diffusion_gpu(ctx, w, h, d)

diff_vol = ones(Float32, w, h, d)
VulkanHeatDiffusion.upload_diffusivity!(state, ctx, diff_vol)
VulkanHeatDiffusion.seed_only!(state, ctx, 32, 32, 32; seed_radius=1)

println("Warmup...")
VulkanHeatDiffusion.run_incremental_steps!(state, ctx, 16; copy_box_to_cpu=true)

println("Benchmarking 100 ticks (1600 steps) WITH CPU readback...")
t0 = time_ns()
for i in 1:100
    VulkanHeatDiffusion.run_incremental_steps!(state, ctx, 16; copy_box_to_cpu=true)
end
t1 = time_ns()
println("Total: ", (t1 - t0) / 1e6, " ms -> per tick: ", (t1 - t0) / 1e8, " ms")

println("Benchmarking 100 ticks WITHOUT CPU readback...")
t2 = time_ns()
for i in 1:100
    VulkanHeatDiffusion.run_incremental_steps!(state, ctx, 16; copy_box_to_cpu=false)
end
t3 = time_ns()
println("Total: ", (t3 - t2) / 1e6, " ms -> per tick: ", (t3 - t2) / 1e8, " ms")

