import re

with open("src/display/Vulkan/VulkanHeatDiffusion.jl", "r") as f:
    code = f.read()

# Update render_slice_to_texture! signature
code = code.replace(
    "function render_slice_to_texture!(state::HeatDiffusionState, ctx, tex::Any, slice_dim::Int, slice_idx::Int, active_id::UInt32; theta::Float32=0.001f0)",
    "function render_slice_to_texture!(state::HeatDiffusionState, ctx, tex::Any, slice_dim::Int, slice_idx::Int, active_id::UInt32, display_step::UInt32; theta::Float32=0.001f0)"
)

code = code.replace(
    "pc_u32 = UInt32[reinterpret(UInt32, theta), UInt32(slice_dim), UInt32(slice_idx), active_id]",
    "pc_u32 = UInt32[reinterpret(UInt32, theta), UInt32(slice_dim), UInt32(slice_idx), active_id, display_step]"
)

with open("src/display/Vulkan/VulkanHeatDiffusion.jl", "w") as f:
    f.write(code)

