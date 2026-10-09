import re
import sys

with open('src/display/Vulkan/VulkanHeatDiffusion.jl', 'r') as f:
    content = f.read()

# 1. Replace Push Constants
old_pc = """# Push constants for extract_mask shader
struct ExtractMaskPC
    theta::Float32
    tau::Float32
    hard_thresh::Int32
    label_id::Int32
end"""

new_pc = """# ─── Push Constants ──────────────────────────────────────────────────────

struct DiffuseStepPC
    dt::Float32
    use_box::Int32
    pad0::Int32
    pad1::Int32
    box_min_x::Int32
    box_min_y::Int32
    box_min_z::Int32
    box_min_w::Int32
    box_max_x::Int32
    box_max_y::Int32
    box_max_z::Int32
    box_max_w::Int32
end

struct ExtractMaskPC
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
end"""

content = content.replace(old_pc, new_pc)

# 2. Add bounding box fields to HeatDiffusionState
old_state_fields = """    # State tracking
    is_initialized::Bool
    diffusivity_uploaded::Bool
    current_buffer_is_ping::Bool   # true = latest data in ping buffer, false = pong
    accumulated_steps::Int         # total diffusion steps since last seed
    
    # Dummy sampler for DescriptorImageInfo"""

new_state_fields = """    # State tracking
    is_initialized::Bool
    diffusivity_uploaded::Bool
    current_buffer_is_ping::Bool   # true = latest data in ping buffer, false = pong
    accumulated_steps::Int         # total diffusion steps since last seed
    
    # Persistent staging buffer for zero-allocation readbacks
    staging_box_buf::Union{Nothing, Buffer}
    staging_box_mem::Union{Nothing, DeviceMemory}
    staging_box_ptr::Ptr{Cvoid}
    staging_box_capacity::Int
    
    # Lesion tracking
    seed_x::Int
    seed_y::Int
    seed_z::Int
    seed_radius::Int
    current_box_min::NTuple{3, Int}
    current_box_max::NTuple{3, Int}
    
    # Dummy sampler for DescriptorImageInfo"""

content = content.replace(old_state_fields, new_state_fields)

# 3. Update HeatDiffusionState constructor
old_constructor = """        state.is_initialized = false
        state.diffusivity_uploaded = false
        state.current_buffer_is_ping = true
        state.accumulated_steps = 0
        state.width = 0
        state.height = 0
        state.depth = 0"""

new_constructor = """        state.is_initialized = false
        state.diffusivity_uploaded = false
        state.current_buffer_is_ping = true
        state.accumulated_steps = 0
        state.width = 0
        state.height = 0
        state.depth = 0
        
        state.staging_box_buf = nothing
        state.staging_box_mem = nothing
        state.staging_box_ptr = C_NULL
        state.staging_box_capacity = 0
        
        state.seed_x = 0
        state.seed_y = 0
        state.seed_z = 0
        state.seed_radius = 0
        state.current_box_min = (1, 1, 1)
        state.current_box_max = (1, 1, 1)"""

content = content.replace(old_constructor, new_constructor)

# 4. Update init_heat_diffusion! (PushConstantRanges and Staging Box allocation)
old_pc_ranges = """    # diffuse_step: push float dt (4 bytes)
    pc_diffuse = [PushConstantRange(SHADER_STAGE_COMPUTE_BIT, 0, 4)]
    state.layout_diffuse_step = unwrap(create_pipeline_layout(dev, PipelineLayoutCreateInfo([state.dsl_diffuse_step], pc_diffuse)))
    
    # extract_mask: push float theta + float tau + int hard + int label_id (16 bytes)
    pc_extract = [PushConstantRange(SHADER_STAGE_COMPUTE_BIT, 0, 16)]
    state.layout_extract_mask = unwrap(create_pipeline_layout(dev, PipelineLayoutCreateInfo([state.dsl_extract_mask], pc_extract)))"""

new_pc_ranges = """    # diffuse_step: push float dt + box... (48 bytes)
    pc_diffuse = [PushConstantRange(SHADER_STAGE_COMPUTE_BIT, 0, 48)]
    state.layout_diffuse_step = unwrap(create_pipeline_layout(dev, PipelineLayoutCreateInfo([state.dsl_diffuse_step], pc_diffuse)))
    
    # extract_mask: push float theta + tau + box... (48 bytes)
    pc_extract = [PushConstantRange(SHADER_STAGE_COMPUTE_BIT, 0, 48)]
    state.layout_extract_mask = unwrap(create_pipeline_layout(dev, PipelineLayoutCreateInfo([state.dsl_extract_mask], pc_extract)))"""

content = content.replace(old_pc_ranges, new_pc_ranges)

# Insert staging box allocation at end of init
old_init_end = """    state.is_initialized = true
    @info "[HeatDiffusion] Initialization complete."
    return state"""

new_init_end = """    # Allocate 16 MB persistent staging box buffer
    box_buf_size = 16 * 1024 * 1024
    buf_info = BufferCreateInfo(box_buf_size, BUFFER_USAGE_TRANSFER_DST_BIT, SHARING_MODE_EXCLUSIVE, UInt32[])
    state.staging_box_buf = unwrap(create_buffer(dev, buf_info))
    
    mem_reqs = get_buffer_memory_requirements(dev, state.staging_box_buf)
    mem_props = get_physical_device_memory_properties(pdev)
    mem_type = UInt32(0)
    found_type = false
    for i in 0:(mem_props.memory_type_count - 1)
        if (mem_reqs.memory_type_bits & (1 << i)) != 0 &&
           (mem_props.memory_types[i + 1].property_flags & (MEMORY_PROPERTY_HOST_VISIBLE_BIT | MEMORY_PROPERTY_HOST_COHERENT_BIT)) == (MEMORY_PROPERTY_HOST_VISIBLE_BIT | MEMORY_PROPERTY_HOST_COHERENT_BIT)
            mem_type = UInt32(i)
            found_type = true
            break
        end
    end
    if !found_type
        error("Failed to find HOST_VISIBLE | HOST_COHERENT memory for staging box buffer")
    end
    
    state.staging_box_mem = unwrap(allocate_memory(dev, MemoryAllocateInfo(mem_reqs.size, mem_type)))
    unwrap(bind_buffer_memory(dev, state.staging_box_buf, state.staging_box_mem, 0))
    state.staging_box_ptr = unwrap(map_memory(dev, state.staging_box_mem, 0, box_buf_size))
    state.staging_box_capacity = box_buf_size

    state.is_initialized = true
    @info "[HeatDiffusion] Initialization complete (allocated 16 MB persistent staging buffer)."
    return state"""

content = content.replace(old_init_end, new_init_end)

# 5. Update destroy_heat_diffusion!
old_destroy = """    state.is_initialized = false
    state.diffusivity_uploaded = false
    @info "[HeatDiffusion] Resources marked for destruction."
end"""

new_destroy = """    state.is_initialized = false
    state.diffusivity_uploaded = false
    if state.staging_box_mem !== nothing && state.staging_box_ptr != C_NULL
        try
            unmap_memory(ctx.device, state.staging_box_mem)
        catch
        end
        state.staging_box_ptr = C_NULL
    end
    @info "[HeatDiffusion] Resources marked for destruction."
end"""

content = content.replace(old_destroy, new_destroy)

# Write it out
with open('src/display/Vulkan/VulkanHeatDiffusion.jl', 'w') as f:
    f.write(content)

print("Applied phase 1 patches successfully")
