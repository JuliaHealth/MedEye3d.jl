"""
    VulkanHeatDiffusion

GPU compute module that runs the heat diffusion PDE entirely on the Vulkan GPU.
This enables sub-millisecond interactive segmentation by avoiding any CPU↔GPU roundtrips
during the user's click-and-hold interaction.

## Pipeline:
1. `init_heat_diffusion!` — Creates compute pipelines, allocates 3D storage images
2. `upload_diffusivity!` — Uploads the precomputed D(x) field to GPU
3. `run_heat_diffusion!` — Seeds + runs K diffusion steps + extracts mask
4. `read_mask_to_cpu` — Async reads the mask back to CPU for saving
5. `destroy_heat_diffusion!` — Cleans up all resources
"""
module VulkanHeatDiffusion

using Vulkan
using VulkanCore
using Logging

export HeatDiffusionState, init_heat_diffusion!, destroy_heat_diffusion!
export upload_diffusivity!, run_heat_diffusion!, read_mask_to_cpu
export seed_only!, run_incremental_steps!

# ─── Push Constants ──────────────────────────────────────────────────────

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
end

# ─── State struct ────────────────────────────────────────────────────────

"""
    HeatDiffusionState

Holds all Vulkan resources for the heat diffusion compute pipeline.
"""
mutable struct HeatDiffusionState
    # Volume dimensions
    width::Int
    height::Int
    depth::Int
    
    # 3D storage images (VK_FORMAT_R32_SFLOAT)
    img_diffusivity::Image          # Precomputed D(x) field
    img_u_ping::Image               # Heat field buffer A
    img_u_pong::Image               # Heat field buffer B
    img_mask::Image                 # Output binary mask
    
    # Image views
    view_diffusivity::ImageView
    view_u_ping::ImageView
    view_u_pong::ImageView
    view_mask::ImageView
    
    # Device memory allocations
    mem_diffusivity::DeviceMemory
    mem_u_ping::DeviceMemory
    mem_u_pong::DeviceMemory
    mem_mask::DeviceMemory
    
    # Compute pipelines
    pipeline_seed_init::Pipeline
    pipeline_diffuse_step::Pipeline
    pipeline_extract_mask::Pipeline
    pipeline_diffusivity::Pipeline
    pipeline_slice::Pipeline
    
    # Pipeline layouts
    layout_seed_init::PipelineLayout
    layout_diffuse_step::PipelineLayout
    layout_extract_mask::PipelineLayout
    layout_diffusivity::PipelineLayout
    layout_slice::PipelineLayout
    
    # Descriptor set layouts
    dsl_seed_init::DescriptorSetLayout
    dsl_diffuse_step::DescriptorSetLayout
    dsl_extract_mask::DescriptorSetLayout
    dsl_diffusivity::DescriptorSetLayout
    
    # Descriptor pool and sets
    descriptor_pool::DescriptorPool
    ds_seed_ping::DescriptorSet          # seed_init writing to ping
    ds_seed_pong::DescriptorSet          # seed_init writing to pong
    ds_diffuse_ping_to_pong::DescriptorSet  # diffuse: read ping, write pong
    ds_diffuse_pong_to_ping::DescriptorSet  # diffuse: read pong, write ping
    ds_extract_from_ping::DescriptorSet     # extract mask from ping
    ds_extract_from_pong::DescriptorSet     # extract mask from pong
    ds_compute_diffusivity::DescriptorSet   # edge→diffusivity conversion
    
    # Compute command pool and buffer
    compute_cmd_pool::CommandPool
    compute_cmd_buffer::CommandBuffer
    compute_fence::Fence
    
    # State tracking
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
    
    # Dummy sampler for DescriptorImageInfo
    dummy_sampler::Sampler
    
    # Default constructor with uninitialized fields
    function HeatDiffusionState()
        state = new()
        state.is_initialized = false
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
        state.current_box_max = (1, 1, 1)
        return state
    end
end

# ─── Shader loading ──────────────────────────────────────────────────────

const SHADER_DIR = joinpath(@__DIR__, "shaders")

function load_shader_module(device::Device, filename::String)
    spv_path = joinpath(SHADER_DIR, filename)
    if !isfile(spv_path)
        error("SPIR-V shader not found: $spv_path")
    end
    code = read(spv_path)
    # Vulkan expects UInt32 aligned code
    code_u32 = reinterpret(UInt32, code)
    create_info = ShaderModuleCreateInfo(sizeof(code), code_u32)
    shader_mod = unwrap(create_shader_module(device, create_info))
    return shader_mod
end

# ─── 3D Image creation helpers ───────────────────────────────────────────

function create_storage_image_3d(device::Device, pdev::PhysicalDevice, w::Int, h::Int, d::Int;
                                  usage_flags = IMAGE_USAGE_STORAGE_BIT | IMAGE_USAGE_TRANSFER_SRC_BIT | IMAGE_USAGE_TRANSFER_DST_BIT)
    create_info = ImageCreateInfo(
        IMAGE_TYPE_3D,
        FORMAT_R32_SFLOAT,
        Extent3D(w, h, d),
        1,  # mip levels
        1,  # array layers
        SAMPLE_COUNT_1_BIT,
        IMAGE_TILING_OPTIMAL,
        usage_flags,
        SHARING_MODE_EXCLUSIVE,
        UInt32[],  # queue family indices
        IMAGE_LAYOUT_UNDEFINED
    )
    img = unwrap(create_image(device, create_info))
    
    # Allocate device memory
    mem_reqs = get_image_memory_requirements(device, img)
    mem_props = get_physical_device_memory_properties(pdev)
    
    mem_type_idx = UInt32(0)
    found = false
    for i in 0:(mem_props.memory_type_count - 1)
        if (mem_reqs.memory_type_bits & (1 << i)) != 0 &&
           (mem_props.memory_types[i + 1].property_flags & MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0
            mem_type_idx = UInt32(i)
            found = true
            break
        end
    end
    if !found
        error("Failed to find suitable memory type for 3D storage image")
    end
    
    alloc_info = MemoryAllocateInfo(mem_reqs.size, mem_type_idx)
    mem = unwrap(allocate_memory(device, alloc_info))
    unwrap(bind_image_memory(device, img, mem, 0))
    
    # Create image view
    view_info = ImageViewCreateInfo(
        img,
        IMAGE_VIEW_TYPE_3D,
        FORMAT_R32_SFLOAT,
        ComponentMapping(
            COMPONENT_SWIZZLE_IDENTITY,
            COMPONENT_SWIZZLE_IDENTITY,
            COMPONENT_SWIZZLE_IDENTITY,
            COMPONENT_SWIZZLE_IDENTITY
        ),
        ImageSubresourceRange(IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1)
    )
    view = unwrap(create_image_view(device, view_info))
    
    return img, mem, view
end

# ─── Initialization ─────────────────────────────────────────────────────

"""
    init_heat_diffusion!(state, ctx, w, h, d)

Initialize all Vulkan resources for heat diffusion on a volume of size (w, h, d).
`ctx` must have fields: `device`, `physical_device`, `graphics_queue`, `queue_family_index`.
"""
function init_heat_diffusion!(state::HeatDiffusionState, ctx, w::Int, h::Int, d::Int)
    if state.is_initialized
        destroy_heat_diffusion!(state, ctx)
    end
    
    state.width = w
    state.height = h
    state.depth = d
    
    dev = ctx.device
    pdev = ctx.physical_device
    
    @info "[HeatDiffusion] Initializing for volume $(w)×$(h)×$(d)..."
    
    # 1. Create 3D storage images
    state.img_diffusivity, state.mem_diffusivity, state.view_diffusivity = create_storage_image_3d(dev, pdev, w, h, d)
    state.img_u_ping, state.mem_u_ping, state.view_u_ping = create_storage_image_3d(dev, pdev, w, h, d)
    state.img_u_pong, state.mem_u_pong, state.view_u_pong = create_storage_image_3d(dev, pdev, w, h, d)
    state.img_mask, state.mem_mask, state.view_mask = create_storage_image_3d(dev, pdev, w, h, d)
    
    # 2. Load shader modules
    mod_seed = load_shader_module(dev, "seed_init.spv")
    mod_diffuse = load_shader_module(dev, "diffuse_step.spv")
    mod_extract = load_shader_module(dev, "extract_mask.spv")
    mod_diff = load_shader_module(dev, "diffusivity.spv")
    
    # 3. Create descriptor set layouts
    # seed_init: 1 storage image (writeonly)
    bindings_seed = [DescriptorSetLayoutBinding(0, DESCRIPTOR_TYPE_STORAGE_IMAGE, SHADER_STAGE_COMPUTE_BIT; descriptor_count=1)]
    state.dsl_seed_init = unwrap(create_descriptor_set_layout(dev, DescriptorSetLayoutCreateInfo(bindings_seed)))
    
    # diffuse_step: 3 storage images (D readonly, u_in readonly, u_out writeonly)
    bindings_diffuse = [
        DescriptorSetLayoutBinding(0, DESCRIPTOR_TYPE_STORAGE_IMAGE, SHADER_STAGE_COMPUTE_BIT; descriptor_count=1),
        DescriptorSetLayoutBinding(1, DESCRIPTOR_TYPE_STORAGE_IMAGE, SHADER_STAGE_COMPUTE_BIT; descriptor_count=1),
        DescriptorSetLayoutBinding(2, DESCRIPTOR_TYPE_STORAGE_IMAGE, SHADER_STAGE_COMPUTE_BIT; descriptor_count=1)
    ]
    state.dsl_diffuse_step = unwrap(create_descriptor_set_layout(dev, DescriptorSetLayoutCreateInfo(bindings_diffuse)))
    
    # extract_mask: 2 storage images (u readonly, mask writeonly)
    bindings_extract = [
        DescriptorSetLayoutBinding(0, DESCRIPTOR_TYPE_STORAGE_IMAGE, SHADER_STAGE_COMPUTE_BIT; descriptor_count=1),
        DescriptorSetLayoutBinding(1, DESCRIPTOR_TYPE_STORAGE_IMAGE, SHADER_STAGE_COMPUTE_BIT; descriptor_count=1)
    ]
    state.dsl_extract_mask = unwrap(create_descriptor_set_layout(dev, DescriptorSetLayoutCreateInfo(bindings_extract)))
    
    # diffusivity: 2 storage images (edge readonly, diffusivity writeonly)
    state.dsl_diffusivity = unwrap(create_descriptor_set_layout(dev, DescriptorSetLayoutCreateInfo(bindings_extract)))
    
    # 4. Create pipeline layouts with push constants
    # seed_init: push ivec4 (16 bytes)
    pc_seed = [PushConstantRange(SHADER_STAGE_COMPUTE_BIT, 0, 16)]
    state.layout_seed_init = unwrap(create_pipeline_layout(dev, PipelineLayoutCreateInfo([state.dsl_seed_init], pc_seed)))
    
    # diffuse_step: push float dt + box... (48 bytes)
    pc_diffuse = [PushConstantRange(SHADER_STAGE_COMPUTE_BIT, 0, 48)]
    state.layout_diffuse_step = unwrap(create_pipeline_layout(dev, PipelineLayoutCreateInfo([state.dsl_diffuse_step], pc_diffuse)))
    
    # extract_mask: push float theta + tau + box... (48 bytes)
    pc_extract = [PushConstantRange(SHADER_STAGE_COMPUTE_BIT, 0, 48)]
    state.layout_extract_mask = unwrap(create_pipeline_layout(dev, PipelineLayoutCreateInfo([state.dsl_extract_mask], pc_extract)))
    
    # diffusivity: push float epsilon_D + float gamma (8 bytes)
    pc_diff = [PushConstantRange(SHADER_STAGE_COMPUTE_BIT, 0, 8)]
    state.layout_diffusivity = unwrap(create_pipeline_layout(dev, PipelineLayoutCreateInfo([state.dsl_diffusivity], pc_diff)))
    
    # 5. Create compute pipelines
    stage_seed = PipelineShaderStageCreateInfo(SHADER_STAGE_COMPUTE_BIT, mod_seed, "main")
    state.pipeline_seed_init = unwrap(create_compute_pipelines(dev, [ComputePipelineCreateInfo(stage_seed, state.layout_seed_init, -1)]))[1][1]
    
    stage_diffuse = PipelineShaderStageCreateInfo(SHADER_STAGE_COMPUTE_BIT, mod_diffuse, "main")
    state.pipeline_diffuse_step = unwrap(create_compute_pipelines(dev, [ComputePipelineCreateInfo(stage_diffuse, state.layout_diffuse_step, -1)]))[1][1]
    
    stage_extract = PipelineShaderStageCreateInfo(SHADER_STAGE_COMPUTE_BIT, mod_extract, "main")
    state.pipeline_extract_mask = unwrap(create_compute_pipelines(dev, [ComputePipelineCreateInfo(stage_extract, state.layout_extract_mask, -1)]))[1][1]
    
    stage_diff = PipelineShaderStageCreateInfo(SHADER_STAGE_COMPUTE_BIT, mod_diff, "main")
    state.pipeline_diffusivity = unwrap(create_compute_pipelines(dev, [ComputePipelineCreateInfo(stage_diff, state.layout_diffusivity, -1)]))[1][1]
    
    # 6. Shader modules are garbage collected automatically by Vulkan.jl
    # destroy_shader_module(dev, mod_seed)
    # destroy_shader_module(dev, mod_diffuse)
    # destroy_shader_module(dev, mod_extract)
    # destroy_shader_module(dev, mod_diff)
    
    # 7. Create descriptor pool and allocate descriptor sets
    pool_sizes = [
        DescriptorPoolSize(DESCRIPTOR_TYPE_STORAGE_IMAGE, 20)  # plenty for all sets
    ]
    state.descriptor_pool = unwrap(create_descriptor_pool(dev, DescriptorPoolCreateInfo(8, pool_sizes)))
    
    # Create dummy sampler for DescriptorImageInfo
    sampler_info = SamplerCreateInfo(
        FILTER_NEAREST, FILTER_NEAREST, SAMPLER_MIPMAP_MODE_NEAREST,
        SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        0.0f0, false, 1.0f0, false, COMPARE_OP_NEVER, 0.0f0, 0.0f0, BORDER_COLOR_FLOAT_TRANSPARENT_BLACK, false
    )
    state.dummy_sampler = unwrap(create_sampler(dev, sampler_info))
    
    # Allocate all descriptor sets
    alloc_ds = DescriptorSetAllocateInfo(state.descriptor_pool, [
        state.dsl_seed_init,       # ds_seed_ping
        state.dsl_seed_init,       # ds_seed_pong
        state.dsl_diffuse_step,    # ds_diffuse_ping_to_pong
        state.dsl_diffuse_step,    # ds_diffuse_pong_to_ping
        state.dsl_extract_mask,    # ds_extract_from_ping
        state.dsl_extract_mask,    # ds_extract_from_pong
        state.dsl_diffusivity      # ds_compute_diffusivity
    ])
    sets = unwrap(Vulkan.allocate_descriptor_sets(dev, alloc_ds))
    state.ds_seed_ping = sets[1]
    state.ds_seed_pong = sets[2]
    state.ds_diffuse_ping_to_pong = sets[3]
    state.ds_diffuse_pong_to_ping = sets[4]
    state.ds_extract_from_ping = sets[5]
    state.ds_extract_from_pong = sets[6]
    state.ds_compute_diffusivity = sets[7]
    
    # 8. Write descriptor sets
    _write_descriptors!(state, dev)
    
    # 9. Create compute command pool & buffer
    pool_info = CommandPoolCreateInfo(ctx.queue_family_index; flags=COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT)
    state.compute_cmd_pool = unwrap(create_command_pool(dev, pool_info))
    
    alloc_cmd = CommandBufferAllocateInfo(state.compute_cmd_pool, COMMAND_BUFFER_LEVEL_PRIMARY, 1)
    state.compute_cmd_buffer = unwrap(Vulkan.allocate_command_buffers(dev, alloc_cmd))[1]
    
    state.compute_fence = unwrap(create_fence(dev, FenceCreateInfo()))
    
    # Allocate 16 MB persistent staging box buffer
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
    return state
end

function _write_descriptors!(state::HeatDiffusionState, dev::Device)
    # Helper to create a DescriptorImageInfo for a storage image
    di(view) = DescriptorImageInfo(state.dummy_sampler, view, IMAGE_LAYOUT_GENERAL)
    
    writes = WriteDescriptorSet[]
    
    # seed_init → ping
    push!(writes, WriteDescriptorSet(state.ds_seed_ping, 0, 0, DESCRIPTOR_TYPE_STORAGE_IMAGE, [di(state.view_u_ping)], [], []))
    # seed_init → pong
    push!(writes, WriteDescriptorSet(state.ds_seed_pong, 0, 0, DESCRIPTOR_TYPE_STORAGE_IMAGE, [di(state.view_u_pong)], [], []))
    
    # diffuse: ping→pong (D, ping_in, pong_out)
    push!(writes, WriteDescriptorSet(state.ds_diffuse_ping_to_pong, 0, 0, DESCRIPTOR_TYPE_STORAGE_IMAGE, [di(state.view_diffusivity)], [], []))
    push!(writes, WriteDescriptorSet(state.ds_diffuse_ping_to_pong, 1, 0, DESCRIPTOR_TYPE_STORAGE_IMAGE, [di(state.view_u_ping)], [], []))
    push!(writes, WriteDescriptorSet(state.ds_diffuse_ping_to_pong, 2, 0, DESCRIPTOR_TYPE_STORAGE_IMAGE, [di(state.view_u_pong)], [], []))
    
    # diffuse: pong→ping (D, pong_in, ping_out)
    push!(writes, WriteDescriptorSet(state.ds_diffuse_pong_to_ping, 0, 0, DESCRIPTOR_TYPE_STORAGE_IMAGE, [di(state.view_diffusivity)], [], []))
    push!(writes, WriteDescriptorSet(state.ds_diffuse_pong_to_ping, 1, 0, DESCRIPTOR_TYPE_STORAGE_IMAGE, [di(state.view_u_pong)], [], []))
    push!(writes, WriteDescriptorSet(state.ds_diffuse_pong_to_ping, 2, 0, DESCRIPTOR_TYPE_STORAGE_IMAGE, [di(state.view_u_ping)], [], []))
    
    # extract from ping (u_ping, mask_out)
    push!(writes, WriteDescriptorSet(state.ds_extract_from_ping, 0, 0, DESCRIPTOR_TYPE_STORAGE_IMAGE, [di(state.view_u_ping)], [], []))
    push!(writes, WriteDescriptorSet(state.ds_extract_from_ping, 1, 0, DESCRIPTOR_TYPE_STORAGE_IMAGE, [di(state.view_mask)], [], []))
    
    # extract from pong (u_pong, mask_out)
    push!(writes, WriteDescriptorSet(state.ds_extract_from_pong, 0, 0, DESCRIPTOR_TYPE_STORAGE_IMAGE, [di(state.view_u_pong)], [], []))
    push!(writes, WriteDescriptorSet(state.ds_extract_from_pong, 1, 0, DESCRIPTOR_TYPE_STORAGE_IMAGE, [di(state.view_mask)], [], []))
    
    # diffusivity compute: edge_in → diffusivity_out
    push!(writes, WriteDescriptorSet(state.ds_compute_diffusivity, 0, 0, DESCRIPTOR_TYPE_STORAGE_IMAGE, [di(state.view_u_ping)], [], []))  # reuse ping for edge input
    push!(writes, WriteDescriptorSet(state.ds_compute_diffusivity, 1, 0, DESCRIPTOR_TYPE_STORAGE_IMAGE, [di(state.view_diffusivity)], [], []))
    
    update_descriptor_sets(dev, writes, [])
end

# ─── Upload diffusivity field ────────────────────────────────────────────

"""
    upload_diffusivity!(state, ctx, diffusivity_data::Array{Float32, 3})

Uploads the precomputed diffusivity field D(x) to the GPU.
`diffusivity_data` must be size (w, h, d).
"""
function upload_diffusivity!(state::HeatDiffusionState, ctx, diffusivity_data::Array{Float32, 3})
    @assert state.is_initialized "Heat diffusion not initialized"
    @assert size(diffusivity_data) == (state.width, state.height, state.depth) "Diffusivity size mismatch"
    
    dev = ctx.device
    pdev = ctx.physical_device
    w, h, d = state.width, state.height, state.depth
    data_size = sizeof(diffusivity_data)
    
    @info "[HeatDiffusion] Uploading diffusivity field ($(w)×$(h)×$(d), $(data_size ÷ 1024) KB)..."
    
    # Create staging buffer
    buf_info = BufferCreateInfo(data_size, BUFFER_USAGE_TRANSFER_SRC_BIT, SHARING_MODE_EXCLUSIVE, UInt32[])
    staging_buf = unwrap(create_buffer(dev, buf_info))
    
    mem_reqs = get_buffer_memory_requirements(dev, staging_buf)
    mem_props = get_physical_device_memory_properties(pdev)
    mem_type = UInt32(0)
    for i in 0:(mem_props.memory_type_count - 1)
        if (mem_reqs.memory_type_bits & (1 << i)) != 0 &&
           (mem_props.memory_types[i + 1].property_flags & (MEMORY_PROPERTY_HOST_VISIBLE_BIT | MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
           (MEMORY_PROPERTY_HOST_VISIBLE_BIT | MEMORY_PROPERTY_HOST_COHERENT_BIT)
            mem_type = UInt32(i)
            break
        end
    end
    
    staging_mem = unwrap(allocate_memory(dev, MemoryAllocateInfo(mem_reqs.size, mem_type)))
    unwrap(bind_buffer_memory(dev, staging_buf, staging_mem, 0))
    
    # Map and copy data
    ptr = unwrap(map_memory(dev, staging_mem, 0, data_size))
    unsafe_copyto!(Ptr{Float32}(ptr), pointer(diffusivity_data), length(diffusivity_data))
    unmap_memory(dev, staging_mem)
    
    # Transition image layout and copy
    one_time_submit!(ctx) do cmd
        # Transition diffusivity image to TRANSFER_DST
        barrier = ImageMemoryBarrier(
            ACCESS_NONE, ACCESS_TRANSFER_WRITE_BIT,
            IMAGE_LAYOUT_UNDEFINED, IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            0, 0,  # queue family indices (ignored)
            state.img_diffusivity,
            ImageSubresourceRange(IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1)
        )
        cmd_pipeline_barrier(cmd, MemoryBarrier[], BufferMemoryBarrier[], [barrier];
                             src_stage_mask=PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             dst_stage_mask=PIPELINE_STAGE_TRANSFER_BIT)
        
        # Copy buffer to 3D image
        region = BufferImageCopy(
            0, 0, 0,
            ImageSubresourceLayers(IMAGE_ASPECT_COLOR_BIT, 0, 0, 1),
            Offset3D(0, 0, 0),
            Extent3D(w, h, d)
        )
        cmd_copy_buffer_to_image(cmd, staging_buf, state.img_diffusivity, IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, [region])
        
        # Transition to GENERAL for compute shader access
        barrier2 = ImageMemoryBarrier(
            ACCESS_TRANSFER_WRITE_BIT, ACCESS_SHADER_READ_BIT,
            IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, IMAGE_LAYOUT_GENERAL,
            0, 0,
            state.img_diffusivity,
            ImageSubresourceRange(IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1)
        )
        cmd_pipeline_barrier(cmd, MemoryBarrier[], BufferMemoryBarrier[], [barrier2];
                             src_stage_mask=PIPELINE_STAGE_TRANSFER_BIT,
                             dst_stage_mask=PIPELINE_STAGE_COMPUTE_SHADER_BIT)
    end
    
    # Staging buffer and memory will be cleaned up by Vulkan.jl GC finalizers
    staging_buf = nothing
    staging_mem = nothing
    
    state.diffusivity_uploaded = true
    @info "[HeatDiffusion] Diffusivity upload complete."
end

# ─── Run heat diffusion ─────────────────────────────────────────────────

"""
    run_heat_diffusion!(state, ctx, seed_x, seed_y, seed_z; K=40, dt=0.16f0, theta=0.001f0, tau=0.0001f0)

Runs the complete heat diffusion pipeline on the GPU:
1. Initialize heat field with seed impulse
2. Run K steps of diffusion
3. Extract binary mask

Returns nothing — the mask is stored in `state.img_mask` on the GPU.
Use `read_mask_to_cpu` to retrieve it.
"""
function run_heat_diffusion!(state::HeatDiffusionState, ctx, seed_x::Int, seed_y::Int, seed_z::Int;
                              K::Int=40, dt::Float32=0.16f0, theta::Float32=0.001f0, tau::Float32=0.0001f0)
    @assert state.is_initialized "Heat diffusion not initialized"
    @assert state.diffusivity_uploaded "Diffusivity not uploaded"
    
    dev = ctx.device
    w, h, d = state.width, state.height, state.depth
    
    # Compute dispatch dimensions (ceil division by workgroup size 8)
    gx = cld(w, 8)
    gy = cld(h, 8)
    gz = cld(d, 8)
    
    # Record command buffer
    cmd = state.compute_cmd_buffer
    unwrap(reset_command_buffer(cmd))
    begin_info = CommandBufferBeginInfo(flags = COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT)
    unwrap(begin_command_buffer(cmd, begin_info))
    
    # --- Step 1: Transition ping/pong images to GENERAL ---
    barriers_init = [
        ImageMemoryBarrier(C_NULL, ACCESS_NONE, ACCESS_SHADER_WRITE_BIT,
            IMAGE_LAYOUT_UNDEFINED, IMAGE_LAYOUT_GENERAL, 0, 0,
            state.img_u_ping, ImageSubresourceRange(IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1)),
        ImageMemoryBarrier(C_NULL, ACCESS_NONE, ACCESS_SHADER_WRITE_BIT,
            IMAGE_LAYOUT_UNDEFINED, IMAGE_LAYOUT_GENERAL, 0, 0,
            state.img_u_pong, ImageSubresourceRange(IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1)),
        ImageMemoryBarrier(C_NULL, ACCESS_NONE, ACCESS_SHADER_WRITE_BIT,
            IMAGE_LAYOUT_UNDEFINED, IMAGE_LAYOUT_GENERAL, 0, 0,
            state.img_mask, ImageSubresourceRange(IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1))
    ]
    cmd_pipeline_barrier(cmd, MemoryBarrier[], BufferMemoryBarrier[], barriers_init;
                         src_stage_mask=PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         dst_stage_mask=PIPELINE_STAGE_COMPUTE_SHADER_BIT)
    
    # --- Step 2: Initialize seed in ping buffer ---
    cmd_bind_pipeline(cmd, PIPELINE_BIND_POINT_COMPUTE, state.pipeline_seed_init)
    cmd_bind_descriptor_sets(cmd, PIPELINE_BIND_POINT_COMPUTE, state.layout_seed_init, 0, [state.ds_seed_ping], [])
    
    seed_data = Int32[seed_x - 1, seed_y - 1, seed_z - 1, 0]
    GC.@preserve seed_data begin
        cmd_push_constants(cmd, state.layout_seed_init, SHADER_STAGE_COMPUTE_BIT, 0, sizeof(seed_data), Ptr{Cvoid}(pointer(seed_data)))
    end
    cmd_dispatch(cmd, gx, gy, gz)
    
    # Barrier: seed write → diffuse read
    barrier_compute = MemoryBarrier(C_NULL, ACCESS_SHADER_WRITE_BIT, ACCESS_SHADER_READ_BIT)
    cmd_pipeline_barrier(cmd, [barrier_compute], BufferMemoryBarrier[], ImageMemoryBarrier[];
                         src_stage_mask=PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         dst_stage_mask=PIPELINE_STAGE_COMPUTE_SHADER_BIT)
    
    # --- Step 3: Run K diffusion steps (ping-pong) ---
    cmd_bind_pipeline(cmd, PIPELINE_BIND_POINT_COMPUTE, state.pipeline_diffuse_step)
    
    dt_arr = Float32[dt]
    for k in 1:K
        if isodd(k)
            # Read from ping, write to pong
            cmd_bind_descriptor_sets(cmd, PIPELINE_BIND_POINT_COMPUTE, state.layout_diffuse_step, 0, [state.ds_diffuse_ping_to_pong], [])
        else
            # Read from pong, write to ping
            cmd_bind_descriptor_sets(cmd, PIPELINE_BIND_POINT_COMPUTE, state.layout_diffuse_step, 0, [state.ds_diffuse_pong_to_ping], [])
        end
        GC.@preserve dt_arr begin
            cmd_push_constants(cmd, state.layout_diffuse_step, SHADER_STAGE_COMPUTE_BIT, 0, sizeof(dt_arr), Ptr{Cvoid}(pointer(dt_arr)))
        end
        cmd_dispatch(cmd, gx, gy, gz)
        
        # Memory barrier between steps
        cmd_pipeline_barrier(cmd, [barrier_compute], BufferMemoryBarrier[], ImageMemoryBarrier[];
                             src_stage_mask=PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             dst_stage_mask=PIPELINE_STAGE_COMPUTE_SHADER_BIT)
    end
    
    # --- Step 4: Extract mask ---
    cmd_bind_pipeline(cmd, PIPELINE_BIND_POINT_COMPUTE, state.pipeline_extract_mask)
    
    # Determine which buffer has the final result (depends on K parity)
    if isodd(K)
        cmd_bind_descriptor_sets(cmd, PIPELINE_BIND_POINT_COMPUTE, state.layout_extract_mask, 0, [state.ds_extract_from_pong], [])
    else
        cmd_bind_descriptor_sets(cmd, PIPELINE_BIND_POINT_COMPUTE, state.layout_extract_mask, 0, [state.ds_extract_from_ping], [])
    end
    
    extract_data = ExtractMaskPC[ExtractMaskPC(Float32(theta), Float32(tau), Int32(1), Int32(1))]
    GC.@preserve extract_data begin
        cmd_push_constants(cmd, state.layout_extract_mask, SHADER_STAGE_COMPUTE_BIT, 0, sizeof(extract_data), Ptr{Cvoid}(pointer(extract_data)))
    end
    cmd_dispatch(cmd, gx, gy, gz)
    
    # Final barrier
    cmd_pipeline_barrier(cmd, [barrier_compute], BufferMemoryBarrier[], ImageMemoryBarrier[];
                         src_stage_mask=PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         dst_stage_mask=PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT)
    
    unwrap(end_command_buffer(cmd))
    
    # Submit and wait
    submit_info = SubmitInfo([], [], [cmd], [])
    unwrap(reset_fences(dev, [state.compute_fence]))
    unwrap(queue_submit(ctx.graphics_queue, [submit_info]; fence=state.compute_fence))
    unwrap(wait_for_fences(dev, [state.compute_fence], true, typemax(UInt64)))
end

# ─── Read mask back to CPU ───────────────────────────────────────────────

"""
    read_mask_to_cpu(state, ctx) → Array{Float32, 3}

Reads the computed mask from GPU back to CPU memory.
"""
function read_mask_to_cpu(state::HeatDiffusionState, ctx)::Array{Float32, 3}
    @assert state.is_initialized "Heat diffusion not initialized"
    
    dev = ctx.device
    pdev = ctx.physical_device
    w, h, d = state.width, state.height, state.depth
    data_size = w * h * d * sizeof(Float32)
    
    # Create staging buffer for readback
    buf_info = BufferCreateInfo(data_size, BUFFER_USAGE_TRANSFER_DST_BIT, SHARING_MODE_EXCLUSIVE, UInt32[])
    staging_buf = unwrap(create_buffer(dev, buf_info))
    
    mem_reqs = get_buffer_memory_requirements(dev, staging_buf)
    mem_props = get_physical_device_memory_properties(pdev)
    mem_type = UInt32(0)
    for i in 0:(mem_props.memory_type_count - 1)
        if (mem_reqs.memory_type_bits & (1 << i)) != 0 &&
           (mem_props.memory_types[i + 1].property_flags & (MEMORY_PROPERTY_HOST_VISIBLE_BIT | MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
           (MEMORY_PROPERTY_HOST_VISIBLE_BIT | MEMORY_PROPERTY_HOST_COHERENT_BIT)
            mem_type = UInt32(i)
            break
        end
    end
    
    staging_mem = unwrap(allocate_memory(dev, MemoryAllocateInfo(mem_reqs.size, mem_type)))
    unwrap(bind_buffer_memory(dev, staging_buf, staging_mem, 0))
    
    # Copy image to buffer
    one_time_submit!(ctx) do cmd
        barrier = ImageMemoryBarrier(
            ACCESS_SHADER_WRITE_BIT, ACCESS_TRANSFER_READ_BIT,
            IMAGE_LAYOUT_GENERAL, IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            0, 0,
            state.img_mask,
            ImageSubresourceRange(IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1)
        )
        cmd_pipeline_barrier(cmd, MemoryBarrier[], BufferMemoryBarrier[], [barrier];
                             src_stage_mask=PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             dst_stage_mask=PIPELINE_STAGE_TRANSFER_BIT)
        
        region = BufferImageCopy(
            0, 0, 0,
            ImageSubresourceLayers(IMAGE_ASPECT_COLOR_BIT, 0, 0, 1),
            Offset3D(0, 0, 0),
            Extent3D(w, h, d)
        )
        cmd_copy_image_to_buffer(cmd, state.img_mask, IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, staging_buf, [region])
        
        # Transition back to GENERAL
        barrier2 = ImageMemoryBarrier(
            ACCESS_TRANSFER_READ_BIT, ACCESS_SHADER_WRITE_BIT,
            IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, IMAGE_LAYOUT_GENERAL,
            0, 0,
            state.img_mask,
            ImageSubresourceRange(IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1)
        )
        cmd_pipeline_barrier(cmd, MemoryBarrier[], BufferMemoryBarrier[], [barrier2];
                             src_stage_mask=PIPELINE_STAGE_TRANSFER_BIT,
                             dst_stage_mask=PIPELINE_STAGE_COMPUTE_SHADER_BIT)
    end
    
    # Map and read data
    ptr = unwrap(map_memory(dev, staging_mem, 0, data_size))
    result = Array{Float32, 3}(undef, w, h, d)
    unsafe_copyto!(pointer(result), Ptr{Float32}(ptr), length(result))
    unmap_memory(dev, staging_mem)
    
    # Staging buffer and memory will be cleaned up by Vulkan.jl GC finalizers
    staging_buf = nothing
    staging_mem = nothing
    
    return result
end

# ─── Incremental (interactive) diffusion ─────────────────────────────────

"""
    seed_only!(state, ctx, seed_x, seed_y, seed_z)

Initialize the heat field with a single seed point at (seed_x, seed_y, seed_z)
without running any diffusion steps. Call `run_incremental_steps!` afterwards
to add diffusion steps incrementally (e.g., per-frame during a mouse hold).
"""
function seed_only!(state::HeatDiffusionState, ctx, seed_x::Int, seed_y::Int, seed_z::Int; seed_radius::Int=3)
    @assert state.is_initialized "Heat diffusion not initialized"
    @assert state.diffusivity_uploaded "Diffusivity not uploaded"
    
    dev = ctx.device
    w, h, d = state.width, state.height, state.depth
    gx, gy, gz = cld(w, 8), cld(h, 8), cld(d, 8)
    
    cmd = state.compute_cmd_buffer
    unwrap(reset_command_buffer(cmd))
    begin_info = CommandBufferBeginInfo(flags = COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT)
    unwrap(begin_command_buffer(cmd, begin_info))
    
    # Transition ping, pong, mask images to GENERAL layout
    barriers_init = [
        ImageMemoryBarrier(C_NULL, ACCESS_NONE, ACCESS_SHADER_WRITE_BIT,
            IMAGE_LAYOUT_UNDEFINED, IMAGE_LAYOUT_GENERAL, 0, 0,
            state.img_u_ping, ImageSubresourceRange(IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1)),
        ImageMemoryBarrier(C_NULL, ACCESS_NONE, ACCESS_SHADER_WRITE_BIT,
            IMAGE_LAYOUT_UNDEFINED, IMAGE_LAYOUT_GENERAL, 0, 0,
            state.img_u_pong, ImageSubresourceRange(IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1)),
        ImageMemoryBarrier(C_NULL, ACCESS_NONE, ACCESS_SHADER_WRITE_BIT,
            IMAGE_LAYOUT_UNDEFINED, IMAGE_LAYOUT_GENERAL, 0, 0,
            state.img_mask, ImageSubresourceRange(IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1))
    ]
    cmd_pipeline_barrier(cmd, MemoryBarrier[], BufferMemoryBarrier[], barriers_init;
                         src_stage_mask=PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         dst_stage_mask=PIPELINE_STAGE_COMPUTE_SHADER_BIT)
    
    # Clear both ping and pong buffers to 0.0 (fast fixed-function clear)
    clear_color = ClearColorValue((0.0f0, 0.0f0, 0.0f0, 0.0f0))
    subres = ImageSubresourceRange(IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1)
    cmd_clear_color_image(cmd, state.img_u_ping, IMAGE_LAYOUT_GENERAL, clear_color, [subres])
    cmd_clear_color_image(cmd, state.img_u_pong, IMAGE_LAYOUT_GENERAL, clear_color, [subres])
    
    # Barrier after clear
    clear_barrier = MemoryBarrier(C_NULL, ACCESS_TRANSFER_WRITE_BIT, ACCESS_SHADER_WRITE_BIT)
    cmd_pipeline_barrier(cmd, [clear_barrier], BufferMemoryBarrier[], ImageMemoryBarrier[];
                         src_stage_mask=PIPELINE_STAGE_TRANSFER_BIT,
                         dst_stage_mask=PIPELINE_STAGE_COMPUTE_SHADER_BIT)
    
    # Seed init into ping buffer
    cmd_bind_pipeline(cmd, PIPELINE_BIND_POINT_COMPUTE, state.pipeline_seed_init)
    cmd_bind_descriptor_sets(cmd, PIPELINE_BIND_POINT_COMPUTE, state.layout_seed_init, 0, [state.ds_seed_ping], [])
    seed_data = Int32[seed_x - 1, seed_y - 1, seed_z - 1, seed_radius]
    GC.@preserve seed_data begin
        cmd_push_constants(cmd, state.layout_seed_init, SHADER_STAGE_COMPUTE_BIT, 0, sizeof(seed_data), Ptr{Cvoid}(pointer(seed_data)))
    end
    cmd_dispatch(cmd, gx, gy, gz)
    
    barrier_compute = MemoryBarrier(C_NULL, ACCESS_SHADER_WRITE_BIT, ACCESS_SHADER_READ_BIT)
    cmd_pipeline_barrier(cmd, [barrier_compute], BufferMemoryBarrier[], ImageMemoryBarrier[];
                         src_stage_mask=PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         dst_stage_mask=PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT)
    
    unwrap(end_command_buffer(cmd))
    submit_info = SubmitInfo([], [], [cmd], [])
    unwrap(reset_fences(dev, [state.compute_fence]))
    unwrap(queue_submit(ctx.graphics_queue, [submit_info]; fence=state.compute_fence))
    unwrap(wait_for_fences(dev, [state.compute_fence], true, typemax(UInt64)))
    
    state.current_buffer_is_ping = true
    state.accumulated_steps = 0
    @debug "[HeatDiffusion] Seeded at ($seed_x, $seed_y, $seed_z)"
end

"""
    run_incremental_steps!(state, ctx, n_steps; dt=0.16f0, theta=0.001f0, tau=0.0001f0)

Runs `n_steps` additional diffusion steps on the GPU without re-seeding,
continuing from whichever ping/pong buffer currently holds the latest data.
After the diffusion steps, automatically runs the extract_mask shader so
`read_mask_to_cpu` can be called immediately.
"""
function run_incremental_steps!(state::HeatDiffusionState, ctx, n_steps::Int;
                                 dt::Float32=0.16f0, theta::Float32=0.001f0, tau::Float32=0.0001f0, copy_box_to_cpu::Bool=false)
    @assert state.is_initialized "Heat diffusion not initialized"
    @assert state.diffusivity_uploaded "Diffusivity not uploaded"
    n_steps <= 0 && return
    
    dev = ctx.device
    w, h, d = state.width, state.height, state.depth
    
    total_steps = state.accumulated_steps + n_steps
    # Expand bounding box
    R = state.seed_radius + round(Int, 1.2 * total_steps) + 6
    xmin = max(1, state.seed_x - R)
    xmax = min(w, state.seed_x + R)
    ymin = max(1, state.seed_y - R)
    ymax = min(h, state.seed_y + R)
    zmin = max(1, state.seed_z - R)
    zmax = min(d, state.seed_z + R)
    
    # Cap bounding box to fit in 16MB buffer if needed
    bw = xmax - xmin + 1
    bh = ymax - ymin + 1
    bd = zmax - zmin + 1
    if (bw * bh * bd * 4) > state.staging_box_capacity
        bw = bh = bd = round(Int, cbrt(state.staging_box_capacity / 4)) - 1
        xmax = min(w, xmin + bw - 1)
        ymax = min(h, ymin + bh - 1)
        zmax = min(d, zmin + bd - 1)
        bw = xmax - xmin + 1
        bh = ymax - ymin + 1
        bd = zmax - zmin + 1
    end
    
    state.current_box_min = (xmin, ymin, zmin)
    state.current_box_max = (xmax, ymax, zmax)
    
    gx = cld(bw, 8)
    gy = cld(bh, 8)
    gz = cld(bd, 8)
    
    cmd = state.compute_cmd_buffer
    unwrap(reset_command_buffer(cmd))
    begin_info = CommandBufferBeginInfo(flags = COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT)
    unwrap(begin_command_buffer(cmd, begin_info))
    
    barrier_compute = MemoryBarrier(C_NULL, ACCESS_SHADER_WRITE_BIT, ACCESS_SHADER_READ_BIT)
    
    # Push constants for bounding box
    box_min_glsl = Int32[xmin - 1, ymin - 1, zmin - 1, 0]
    box_max_glsl = Int32[xmax - 1, ymax - 1, zmax - 1, 0]
    
    pc_diffuse = DiffuseStepPC(
        dt, Int32(1), Int32(0), Int32(0),
        box_min_glsl[1], box_min_glsl[2], box_min_glsl[3], Int32(0),
        box_max_glsl[1], box_max_glsl[2], box_max_glsl[3], Int32(0)
    )
    
    # ── Diffusion steps (continue from current ping/pong state) ──
    cmd_bind_pipeline(cmd, PIPELINE_BIND_POINT_COMPUTE, state.pipeline_diffuse_step)
    
    is_ping = state.current_buffer_is_ping
    for k in 1:n_steps
        if is_ping
            cmd_bind_descriptor_sets(cmd, PIPELINE_BIND_POINT_COMPUTE, state.layout_diffuse_step, 0, [state.ds_diffuse_ping_to_pong], [])
        else
            cmd_bind_descriptor_sets(cmd, PIPELINE_BIND_POINT_COMPUTE, state.layout_diffuse_step, 0, [state.ds_diffuse_pong_to_ping], [])
        end
        GC.@preserve pc_diffuse begin
            cmd_push_constants(cmd, state.layout_diffuse_step, SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc_diffuse), Ptr{Cvoid}(pointer_from_objref(Ref(pc_diffuse))))
        end
        cmd_dispatch(cmd, gx, gy, gz)
        cmd_pipeline_barrier(cmd, [barrier_compute], BufferMemoryBarrier[], ImageMemoryBarrier[];
                             src_stage_mask=PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             dst_stage_mask=PIPELINE_STAGE_COMPUTE_SHADER_BIT)
        is_ping = !is_ping
    end
    
    # ── Extract mask ──
    cmd_bind_pipeline(cmd, PIPELINE_BIND_POINT_COMPUTE, state.pipeline_extract_mask)
    ds_extract = is_ping ? state.ds_extract_from_ping : state.ds_extract_from_pong
    cmd_bind_descriptor_sets(cmd, PIPELINE_BIND_POINT_COMPUTE, state.layout_extract_mask, 0, [ds_extract], [])
    
    pc_extract = ExtractMaskPC(
        theta, tau, Int32(1), Int32(1),
        box_min_glsl[1], box_min_glsl[2], box_min_glsl[3], Int32(0),
        box_max_glsl[1], box_max_glsl[2], box_max_glsl[3], Int32(0)
    )
    GC.@preserve pc_extract begin
        cmd_push_constants(cmd, state.layout_extract_mask, SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc_extract), Ptr{Cvoid}(pointer_from_objref(Ref(pc_extract))))
    end
    cmd_dispatch(cmd, gx, gy, gz)
    
    if copy_box_to_cpu && state.staging_box_buf !== nothing
        barrier_comp_to_xfer = MemoryBarrier(C_NULL, ACCESS_SHADER_WRITE_BIT, ACCESS_TRANSFER_READ_BIT)
        cmd_pipeline_barrier(cmd, [barrier_comp_to_xfer], BufferMemoryBarrier[], ImageMemoryBarrier[];
                             src_stage_mask=PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             dst_stage_mask=PIPELINE_STAGE_TRANSFER_BIT)
        
        region = BufferImageCopy(
            0, 0, 0,
            ImageSubresourceLayers(IMAGE_ASPECT_COLOR_BIT, 0, 0, 1),
            Offset3D(box_min_glsl[1], box_min_glsl[2], box_min_glsl[3]),
            Extent3D(bw, bh, bd)
        )
        cmd_copy_image_to_buffer(cmd, state.img_mask, IMAGE_LAYOUT_GENERAL, state.staging_box_buf, [region])
        
        barrier_xfer_to_host = MemoryBarrier(C_NULL, ACCESS_TRANSFER_WRITE_BIT, ACCESS_HOST_READ_BIT)
        cmd_pipeline_barrier(cmd, [barrier_xfer_to_host], BufferMemoryBarrier[], ImageMemoryBarrier[];
                             src_stage_mask=PIPELINE_STAGE_TRANSFER_BIT,
                             dst_stage_mask=PIPELINE_STAGE_HOST_BIT)
    else
        cmd_pipeline_barrier(cmd, [barrier_compute], BufferMemoryBarrier[], ImageMemoryBarrier[];
                             src_stage_mask=PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             dst_stage_mask=PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT)
    end
    
    unwrap(end_command_buffer(cmd))
    submit_info = SubmitInfo([], [], [cmd], [])
    unwrap(reset_fences(dev, [state.compute_fence]))
    unwrap(queue_submit(ctx.graphics_queue, [submit_info]; fence=state.compute_fence))
    unwrap(wait_for_fences(dev, [state.compute_fence], true, typemax(UInt64)))
    
    state.current_buffer_is_ping = is_ping
    state.accumulated_steps += n_steps
end

# ─── Cleanup ─────────────────────────────────────────────────────────────

function destroy_heat_diffusion!(state::HeatDiffusionState, ctx)
    if !state.is_initialized
        return
    end
    
    # Wait for GPU idle then let Vulkan.jl GC finalizers clean up resources
    # (Manual destroy_*/free_memory causes double-free on GC exit)
    unwrap(device_wait_idle(ctx.device))
    
    state.is_initialized = false
    state.diffusivity_uploaded = false
    if state.staging_box_mem !== nothing && state.staging_box_ptr != C_NULL
        try
            unmap_memory(ctx.device, state.staging_box_mem)
        catch
        end
        state.staging_box_ptr = C_NULL
    end
    @info "[HeatDiffusion] Resources marked for destruction."
end

# ─── one_time_submit! helper (uses the context's graphics queue) ─────────

function one_time_submit!(f::Function, ctx)
    alloc_info = CommandBufferAllocateInfo(ctx.command_pool, COMMAND_BUFFER_LEVEL_PRIMARY, 1)
    cmds = unwrap(Vulkan.allocate_command_buffers(ctx.device, alloc_info))
    cmd = cmds[1]
    begin_info = CommandBufferBeginInfo(flags = COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT)
    unwrap(begin_command_buffer(cmd, begin_info))
    f(cmd)
    unwrap(end_command_buffer(cmd))
    submit_info = SubmitInfo([], [], [cmd], [])
    unwrap(queue_submit(ctx.graphics_queue, [submit_info]))
    unwrap(queue_wait_idle(ctx.graphics_queue))
    free_command_buffers(ctx.device, ctx.command_pool, [cmd])
end


# ─── Read bounding box mask directly to seg_vol ──────────────────────────

"""
    apply_box_mask_to_segvol!(state, seg_vol::AbstractArray{T, 3}, label_val::T; theta=0.001f0) where {T}

Applies the bounding-box mask retrieved from GPU directly into `seg_vol`.
Reads from the persistently mapped `staging_box_ptr` with ZERO allocations.
"""
function apply_box_mask_to_segvol!(state::HeatDiffusionState, seg_vol::AbstractArray{T, 3}, label_val::T; theta::Float32=0.001f0) where {T}
    state.staging_box_ptr == C_NULL && return
    
    xmin, ymin, zmin = state.current_box_min
    xmax, ymax, zmax = state.current_box_max
    bw = xmax - xmin + 1
    bh = ymax - ymin + 1
    bd = zmax - zmin + 1
    
    ptr = Ptr{Float32}(state.staging_box_ptr)
    
    @inbounds for kz in 1:bd
        z = zmin + kz - 1
        for ky in 1:bh
            y = ymin + ky - 1
            for kx in 1:bw
                x = xmin + kx - 1
                val = unsafe_load(ptr, kx + (ky - 1) * bw + (kz - 1) * bw * bh)
                if val >= theta
                    seg_vol[x, y, z] = label_val
                end
            end
        end
    end
end

end # module VulkanHeatDiffusion
