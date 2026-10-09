import re

with open("src/display/Vulkan/VulkanHeatDiffusion.jl", "r") as f:
    code = f.read()

# Update bindings_extract
code = code.replace(
    """    # extract_mask: 2 storage images (u readonly, mask writeonly)
    bindings_extract = [
        DescriptorSetLayoutBinding(0, DESCRIPTOR_TYPE_STORAGE_IMAGE, SHADER_STAGE_COMPUTE_BIT; descriptor_count=1),
        DescriptorSetLayoutBinding(1, DESCRIPTOR_TYPE_STORAGE_IMAGE, SHADER_STAGE_COMPUTE_BIT; descriptor_count=1)
    ]""",
    """    # extract_mask: 3 storage images (u readonly, mask writeonly, arrival readonly)
    bindings_extract = [
        DescriptorSetLayoutBinding(0, DESCRIPTOR_TYPE_STORAGE_IMAGE, SHADER_STAGE_COMPUTE_BIT; descriptor_count=1),
        DescriptorSetLayoutBinding(1, DESCRIPTOR_TYPE_STORAGE_IMAGE, SHADER_STAGE_COMPUTE_BIT; descriptor_count=1),
        DescriptorSetLayoutBinding(2, DESCRIPTOR_TYPE_STORAGE_IMAGE, SHADER_STAGE_COMPUTE_BIT; descriptor_count=1)
    ]"""
)

# Update ds_extract_from_ping/pong writes
code = code.replace(
    """    # extract: from ping (ping_in, mask_out)
    push!(writes, WriteDescriptorSet(state.ds_extract_from_ping, 0, 0, DESCRIPTOR_TYPE_STORAGE_IMAGE, [di(state.view_u_ping)], [], []))
    push!(writes, WriteDescriptorSet(state.ds_extract_from_ping, 1, 0, DESCRIPTOR_TYPE_STORAGE_IMAGE, [di(state.view_mask)], [], []))
    
    # extract: from pong (pong_in, mask_out)
    push!(writes, WriteDescriptorSet(state.ds_extract_from_pong, 0, 0, DESCRIPTOR_TYPE_STORAGE_IMAGE, [di(state.view_u_pong)], [], []))
    push!(writes, WriteDescriptorSet(state.ds_extract_from_pong, 1, 0, DESCRIPTOR_TYPE_STORAGE_IMAGE, [di(state.view_mask)], [], []))""",
    """    # extract: from ping (ping_in, mask_out, arrival_in)
    push!(writes, WriteDescriptorSet(state.ds_extract_from_ping, 0, 0, DESCRIPTOR_TYPE_STORAGE_IMAGE, [di(state.view_u_ping)], [], []))
    push!(writes, WriteDescriptorSet(state.ds_extract_from_ping, 1, 0, DESCRIPTOR_TYPE_STORAGE_IMAGE, [di(state.view_mask)], [], []))
    push!(writes, WriteDescriptorSet(state.ds_extract_from_ping, 2, 0, DESCRIPTOR_TYPE_STORAGE_IMAGE, [di(state.view_arrival)], [], []))
    
    # extract: from pong (pong_in, mask_out, arrival_in)
    push!(writes, WriteDescriptorSet(state.ds_extract_from_pong, 0, 0, DESCRIPTOR_TYPE_STORAGE_IMAGE, [di(state.view_u_pong)], [], []))
    push!(writes, WriteDescriptorSet(state.ds_extract_from_pong, 1, 0, DESCRIPTOR_TYPE_STORAGE_IMAGE, [di(state.view_mask)], [], []))
    push!(writes, WriteDescriptorSet(state.ds_extract_from_pong, 2, 0, DESCRIPTOR_TYPE_STORAGE_IMAGE, [di(state.view_arrival)], [], []))"""
)

# Update ExtractMaskPC struct and push constants
code = code.replace(
    """struct ExtractMaskPC
    theta::Float32
    tau::Float32
    hard_thresh::Int32
    use_box::Int32
    box_min_x::Int32
    box_min_y::Int32
    box_min_z::Int32
    box_min_w::Int32
    box_max_x::Int32
    box_max_y::Int32
    box_max_z::Int32
    box_max_w::Int32
end""",
    """struct ExtractMaskPC
    theta::Float32
    tau::Float32
    hard_thresh::Int32
    use_box::Int32
    box_min_x::Int32
    box_min_y::Int32
    box_min_z::Int32
    box_min_w::Int32
    box_max_x::Int32
    box_max_y::Int32
    box_max_z::Int32
    box_max_w::Int32
    display_step::UInt32
end"""
)

code = code.replace(
    "pc_extract = [PushConstantRange(SHADER_STAGE_COMPUTE_BIT, 0, 48)]",
    "pc_extract = [PushConstantRange(SHADER_STAGE_COMPUTE_BIT, 0, 52)]"
)

# Update run_incremental_steps! pc_extract initialization
old_pc_extract = """    pc_extract = ExtractMaskPC(
        theta, tau, Int32(1), Int32(1),
        box_min_glsl[1], box_min_glsl[2], box_min_glsl[3], Int32(0),
        box_max_glsl[1], box_max_glsl[2], box_max_glsl[3], Int32(0)
    )"""
new_pc_extract = """    pc_extract = ExtractMaskPC(
        theta, tau, Int32(2), Int32(1),
        box_min_glsl[1], box_min_glsl[2], box_min_glsl[3], Int32(0),
        box_max_glsl[1], box_max_glsl[2], box_max_glsl[3], Int32(0),
        UInt32(display_step)
    )"""

code = code.replace(
    "function run_incremental_steps!(state::HeatDiffusionState, ctx, n_steps::Int; theta::Float32=0.001f0, tau::Float32=0.0001f0, copy_box_to_cpu::Bool=false)",
    "function run_incremental_steps!(state::HeatDiffusionState, ctx, n_steps::Int; theta::Float32=0.001f0, tau::Float32=0.0001f0, copy_box_to_cpu::Bool=false, display_step::Int=-1)"
)
code = code.replace("display_step::Int=-1", "display_step::Int=0") # Default to 0 to avoid -1 uint wrap? Wait, if display_step is 0, it won't show anything. Wait, let's default to state.accumulated_steps + n_steps
code = code.replace("display_step::Int=0", "display_step::Int=typemax(Int32)") # 2 billion steps is basically infinity
code = code.replace(old_pc_extract, new_pc_extract)

with open("src/display/Vulkan/VulkanHeatDiffusion.jl", "w") as f:
    f.write(code)

