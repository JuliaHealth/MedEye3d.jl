"""
    VulkanPipeline

Manages Vulkan graphics pipelines, descriptor set layouts, pipeline layouts,
descriptor sets, and Uniform Buffer Objects (UBO) for texture parameters.

Replaces OpenGL shader program binding and uniform setting from
`CustomFragShad.jl`, `ShadersAndVerticies.jl`, and `Uniforms.jl`.
"""
module VulkanPipeline

using Vulkan
using VulkanCore
using ..VulkanContext: VkCtx
using ..VulkanBuffers: create_buffer, VkQuadBuffers
using ..VulkanShaders: compile_glsl_to_spirv, create_shader_module
using ..VulkanTextures: VkTexture

export VkPipelineState, create_pipeline_state, destroy_pipeline_state!
export update_ubo!, update_descriptor_textures!

# ─── Constants for std140 UBO layout ───────────────────────────────────
# Matches the std140 uniform block in VulkanShaders.jl:
# Each texture parameter block = 128 bytes:
#   offset   0: int   isVisible (4 bytes)
#   offset   4: float minValue  (4 bytes)
#   offset   8: float maxValue  (4 bytes)
#   offset  12: float ValueRange(4 bytes)
#   offset  16: float maskContribution (4 bytes)
#   offset  20..31: padding (12 bytes)
#   offset  32: vec4  colorMask (16 bytes)
#   offset  48: int   allowedIDCount (4 bytes)
#   offset  52..63: padding (12 bytes)
#   offset  64: float allowedIDs[16] (64 bytes)
# Total = 320 bytes
const TEXTURE_PARAMS_SIZE = 320

"""
    VkPipelineState

Holds all the Vulkan pipeline resources for a rendering pipeline:
- Graphics pipeline handle
- Pipeline layout handle
- Descriptor set layout handles
- Descriptor pool and allocated descriptor sets
- UBO buffer and memory for uniform parameter updates
"""
mutable struct VkPipelineState
    pipeline::Pipeline
    pipeline_layout::PipelineLayout
    descriptor_set_layout_samplers::DescriptorSetLayout
    descriptor_set_layout_ubo::DescriptorSetLayout
    descriptor_pool::DescriptorPool
    descriptor_set_samplers::DescriptorSet
    descriptor_set_ubo::DescriptorSet
    ubo_buffer::Buffer
    ubo_memory::DeviceMemory
    ubo_size::Int
    vert_module::ShaderModule
    frag_module::ShaderModule
    n_textures::Int
    mapped_ubo_ptr::Ptr{UInt8}
    # Dirty tracking: skip redundant UBO writes when nothing changed
    ubo_dirty::Bool
    # Pre-allocated scratch buffer for UBO packing (eliminates per-frame zeros() alloc)
    _ubo_scratch::Vector{UInt8}
end

# ─── Descriptor set layout creation via raw VulkanCore ─────────────────

function create_sampler_dsl(device::Device, n_textures::Int)::DescriptorSetLayout
    bindings = [
        VulkanCore.VkDescriptorSetLayoutBinding(
            UInt32(i - 1),
            VulkanCore.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            UInt32(1),
            VulkanCore.VK_SHADER_STAGE_FRAGMENT_BIT,
            C_NULL
        )
        for i in 1:n_textures
    ]
    dsl_ref = Ref(VulkanCore.VkDescriptorSetLayout(C_NULL))
    GC.@preserve bindings begin
        ci = Ref(VulkanCore.VkDescriptorSetLayoutCreateInfo(
            VulkanCore.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            C_NULL, UInt32(0),
            UInt32(length(bindings)), pointer(bindings)
        ))
        fptr_dsl = Vulkan.function_pointer(Vulkan.global_dispatcher[], device, :vkCreateDescriptorSetLayout)
        res = ccall(fptr_dsl, VulkanCore.VkResult,
            (VulkanCore.VkDevice, Ptr{VulkanCore.VkDescriptorSetLayoutCreateInfo}, Ptr{Cvoid}, Ptr{VulkanCore.VkDescriptorSetLayout}),
            device.vks, ci, C_NULL, dsl_ref)
        res != VulkanCore.VK_SUCCESS && error("vkCreateDescriptorSetLayout failed: $res")
    end
    return DescriptorSetLayout(dsl_ref[], device, Threads.Atomic{UInt64}(1))
end

function create_ubo_dsl(device::Device)::DescriptorSetLayout
    bindings = [
        VulkanCore.VkDescriptorSetLayoutBinding(
            UInt32(0),
            VulkanCore.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            UInt32(1),
            VulkanCore.VK_SHADER_STAGE_FRAGMENT_BIT,
            C_NULL
        )
    ]
    dsl_ref = Ref(VulkanCore.VkDescriptorSetLayout(C_NULL))
    GC.@preserve bindings begin
        ci = Ref(VulkanCore.VkDescriptorSetLayoutCreateInfo(
            VulkanCore.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            C_NULL, UInt32(0),
            UInt32(length(bindings)), pointer(bindings)
        ))
        fptr_dsl = Vulkan.function_pointer(Vulkan.global_dispatcher[], device, :vkCreateDescriptorSetLayout)
        res = ccall(fptr_dsl, VulkanCore.VkResult,
            (VulkanCore.VkDevice, Ptr{VulkanCore.VkDescriptorSetLayoutCreateInfo}, Ptr{Cvoid}, Ptr{VulkanCore.VkDescriptorSetLayout}),
            device.vks, ci, C_NULL, dsl_ref)
        res != VulkanCore.VK_SUCCESS && error("vkCreateDescriptorSetLayout failed: $res")
    end
    return DescriptorSetLayout(dsl_ref[], device, Threads.Atomic{UInt64}(1))
end

# ─── Pipeline creation ──────────────────────────────────────────────────

"""
    create_pipeline_state(ctx, vert_glsl, frag_glsl, n_textures) → VkPipelineState

Compiles shaders, creates descriptor set layouts, pipeline layout,
graphics pipeline, descriptor pool, descriptor sets, and UBO buffer.
"""
function create_pipeline_state(ctx::VkCtx, vert_glsl::String, frag_glsl::String,
                                n_textures::Int)::VkPipelineState
    # Compile shaders
    vert_spv = compile_glsl_to_spirv(:vert, vert_glsl)
    frag_spv = compile_glsl_to_spirv(:frag, frag_glsl)
    vert_mod = create_shader_module(ctx.device, vert_spv)
    frag_mod = create_shader_module(ctx.device, frag_spv)

    # Descriptor set layouts
    dsl_samplers = create_sampler_dsl(ctx.device, n_textures)
    dsl_ubo = create_ubo_dsl(ctx.device)

    # Push constant range (zoom/pan + NDC bounds: 4x vec2 = 32 bytes)
    push_range = VulkanCore.VkPushConstantRange(
        VulkanCore.VK_SHADER_STAGE_VERTEX_BIT,
        UInt32(0),   # offset
        UInt32(32)   # size: uvScale(8) + uvOffset(8) + ndcMin(8) + ndcMax(8)
    )
    push_ranges = [push_range]

    # Pipeline layout — raw VulkanCore
    dsl_handles = [dsl_samplers.vks, dsl_ubo.vks]
    pl_ref = Ref(VulkanCore.VkPipelineLayout(C_NULL))
    GC.@preserve dsl_handles push_ranges begin
        plci = Ref(VulkanCore.VkPipelineLayoutCreateInfo(
            VulkanCore.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            C_NULL,
            UInt32(0),
            UInt32(length(dsl_handles)), pointer(dsl_handles),
            UInt32(length(push_ranges)), pointer(push_ranges)
        ))
        fptr = Vulkan.function_pointer(Vulkan.global_dispatcher[], ctx.device, :vkCreatePipelineLayout)
        result = ccall(fptr, VulkanCore.VkResult,
            (VulkanCore.VkDevice, Ptr{VulkanCore.VkPipelineLayoutCreateInfo}, Ptr{Cvoid}, Ptr{VulkanCore.VkPipelineLayout}),
            ctx.device.vks, plci, C_NULL, pl_ref)
        result != VulkanCore.VK_SUCCESS && error("vkCreatePipelineLayout failed: $result")
    end
    pipeline_layout = PipelineLayout(pl_ref[], ctx.device, Threads.Atomic{UInt64}(1))

    # Graphics pipeline
    pipeline = _create_graphics_pipeline_raw(ctx, vert_mod, frag_mod, pipeline_layout, zero_vbo=true)

    # Descriptor pool
    pool_sizes = [
        DescriptorPoolSize(DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, UInt32(max(n_textures, 1))),
        DescriptorPoolSize(DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1)
    ]
    dpci = DescriptorPoolCreateInfo(2, pool_sizes; flags=DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT)
    pool = unwrap(create_descriptor_pool(ctx.device, dpci))

    # Allocate descriptor sets
    dsai = DescriptorSetAllocateInfo(pool, [dsl_samplers, dsl_ubo])
    desc_sets = unwrap(allocate_descriptor_sets(ctx.device, dsai))

    # UBO buffer
    ubo_size = TEXTURE_PARAMS_SIZE * max(n_textures, 1)
    ubo_buf, ubo_mem = create_buffer(ctx, ubo_size,
        BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        MEMORY_PROPERTY_HOST_VISIBLE_BIT | MEMORY_PROPERTY_HOST_COHERENT_BIT)

    # Bind UBO to descriptor set 1
    ubo_info = DescriptorBufferInfo(ubo_buf, 0, UInt64(ubo_size))
    write_ubo = WriteDescriptorSet(
        desc_sets[2],
        0,
        0,
        DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        [],
        [ubo_info],
        []
    )
    update_descriptor_sets(ctx.device, [write_ubo], [])

    # Map UBO memory once and cache pointer for fast updates
    ptr = unwrap(map_memory(ctx.device, ubo_mem, 0, UInt64(ubo_size)))
    mapped_ubo_ptr = Ptr{UInt8}(ptr)

    return VkPipelineState(
        pipeline, pipeline_layout,
        dsl_samplers, dsl_ubo,
        pool, desc_sets[1], desc_sets[2],
        ubo_buf, ubo_mem, ubo_size,
        vert_mod, frag_mod, n_textures,
        mapped_ubo_ptr,
        true,                      # ubo_dirty — always write on first frame
        zeros(UInt8, ubo_size)     # _ubo_scratch — pre-allocated packing buffer
    )
end

# ─── Low-level pipeline creation via VulkanCore ─────────────────────────

function _create_graphics_pipeline_raw(ctx::VkCtx, vert_mod::ShaderModule,
                                        frag_mod::ShaderModule,
                                        pipeline_layout::PipelineLayout;
                                        zero_vbo::Bool=false)::Pipeline
    entry = b"main\0"

    # Vertex input: position(3) + color(3) + texcoord(2) = stride 32 (or empty for Zero-VBO)
    bindings_arr = zero_vbo ? VulkanCore.VkVertexInputBindingDescription[] : [VulkanCore.VkVertexInputBindingDescription(0, 32, VulkanCore.VK_VERTEX_INPUT_RATE_VERTEX)]
    attrs = zero_vbo ? VulkanCore.VkVertexInputAttributeDescription[] : [
        VulkanCore.VkVertexInputAttributeDescription(0, 0, VulkanCore.VK_FORMAT_R32G32B32_SFLOAT, 0),   # position
        VulkanCore.VkVertexInputAttributeDescription(1, 0, VulkanCore.VK_FORMAT_R32G32B32_SFLOAT, 12),  # color
        VulkanCore.VkVertexInputAttributeDescription(2, 0, VulkanCore.VK_FORMAT_R32G32_SFLOAT, 24),     # texcoord
    ]
    p_bindings = isempty(bindings_arr) ? C_NULL : pointer(bindings_arr)
    p_attrs = isempty(attrs) ? C_NULL : pointer(attrs)

    # Viewport and scissor (dynamic)
    vps = [VulkanCore.VkViewport(0.0f0, 0.0f0, Float32(ctx.width), Float32(ctx.height), 0.0f0, 1.0f0)]
    scs = [VulkanCore.VkRect2D(VulkanCore.VkOffset2D(0, 0), VulkanCore.VkExtent2D(ctx.width, ctx.height))]

    # Blending
    blend_atts = [VulkanCore.VkPipelineColorBlendAttachmentState(
        UInt32(0),
        VulkanCore.VK_BLEND_FACTOR_ONE,
        VulkanCore.VK_BLEND_FACTOR_ZERO,
        VulkanCore.VK_BLEND_OP_ADD,
        VulkanCore.VK_BLEND_FACTOR_ONE,
        VulkanCore.VK_BLEND_FACTOR_ZERO,
        VulkanCore.VK_BLEND_OP_ADD,
        UInt32(0x0f)
    )]

    # Dynamic states
    dyn_states = [VulkanCore.VK_DYNAMIC_STATE_VIEWPORT, VulkanCore.VK_DYNAMIC_STATE_SCISSOR]

    stages = Vector{VulkanCore.VkPipelineShaderStageCreateInfo}(undef, 2)
    vi_arr = Vector{VulkanCore.VkPipelineVertexInputStateCreateInfo}(undef, 1)
    ia_arr = Vector{VulkanCore.VkPipelineInputAssemblyStateCreateInfo}(undef, 1)
    vp_arr = Vector{VulkanCore.VkPipelineViewportStateCreateInfo}(undef, 1)
    raster_arr = Vector{VulkanCore.VkPipelineRasterizationStateCreateInfo}(undef, 1)
    ms_arr = Vector{VulkanCore.VkPipelineMultisampleStateCreateInfo}(undef, 1)
    cb_arr = Vector{VulkanCore.VkPipelineColorBlendStateCreateInfo}(undef, 1)
    dyn_arr = Vector{VulkanCore.VkPipelineDynamicStateCreateInfo}(undef, 1)
    ci_arr = Vector{VulkanCore.VkGraphicsPipelineCreateInfo}(undef, 1)

    pipeline_ref = Ref(VulkanCore.VkPipeline(C_NULL))

    GC.@preserve entry vert_mod frag_mod bindings_arr attrs vps scs blend_atts dyn_states stages vi_arr ia_arr vp_arr raster_arr ms_arr cb_arr dyn_arr ci_arr begin
        stages[1] = VulkanCore.VkPipelineShaderStageCreateInfo(
            VulkanCore.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            C_NULL, UInt32(0),
            VulkanCore.VK_SHADER_STAGE_VERTEX_BIT,
            vert_mod.vks, pointer(entry), C_NULL
        )
        stages[2] = VulkanCore.VkPipelineShaderStageCreateInfo(
            VulkanCore.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            C_NULL, UInt32(0),
            VulkanCore.VK_SHADER_STAGE_FRAGMENT_BIT,
            frag_mod.vks, pointer(entry), C_NULL
        )

        vi_arr[1] = VulkanCore.VkPipelineVertexInputStateCreateInfo(
            VulkanCore.VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            C_NULL, UInt32(0),
            UInt32(length(bindings_arr)), p_bindings,
            UInt32(length(attrs)), p_attrs
        )
        ia_arr[1] = VulkanCore.VkPipelineInputAssemblyStateCreateInfo(
            VulkanCore.VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            C_NULL, UInt32(0),
            VulkanCore.VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, UInt32(0)
        )
        vp_arr[1] = VulkanCore.VkPipelineViewportStateCreateInfo(
            VulkanCore.VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            C_NULL, UInt32(0), UInt32(1), pointer(vps), UInt32(1), pointer(scs)
        )
        raster_arr[1] = VulkanCore.VkPipelineRasterizationStateCreateInfo(
            VulkanCore.VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            C_NULL, UInt32(0),
            UInt32(0), UInt32(0),
            VulkanCore.VK_POLYGON_MODE_FILL,
            UInt32(0),
            VulkanCore.VK_FRONT_FACE_COUNTER_CLOCKWISE,
            UInt32(0), 0.0f0, 0.0f0, 0.0f0, 1.0f0
        )
        ms_arr[1] = VulkanCore.VkPipelineMultisampleStateCreateInfo(
            VulkanCore.VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            C_NULL, UInt32(0),
            VulkanCore.VK_SAMPLE_COUNT_1_BIT,
            UInt32(0), 1.0f0, C_NULL, UInt32(0), UInt32(0)
        )
        cb_arr[1] = VulkanCore.VkPipelineColorBlendStateCreateInfo(
            VulkanCore.VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            C_NULL, UInt32(0),
            UInt32(0), VulkanCore.VK_LOGIC_OP_COPY,
            UInt32(1), pointer(blend_atts),
            (0.0f0, 0.0f0, 0.0f0, 0.0f0)
        )
        dyn_arr[1] = VulkanCore.VkPipelineDynamicStateCreateInfo(
            VulkanCore.VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
            C_NULL, UInt32(0),
            UInt32(2), pointer(dyn_states)
        )

        ci_arr[1] = VulkanCore.VkGraphicsPipelineCreateInfo(
            VulkanCore.VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            C_NULL, UInt32(0),
            UInt32(2), pointer(stages),
            pointer(vi_arr),
            pointer(ia_arr),
            C_NULL,
            pointer(vp_arr),
            pointer(raster_arr),
            pointer(ms_arr),
            C_NULL,
            pointer(cb_arr),
            pointer(dyn_arr),
            pipeline_layout.vks,
            ctx.render_pass.vks,
            UInt32(0),
            C_NULL,
            Int32(-1)
        )

        fptr = Vulkan.function_pointer(Vulkan.global_dispatcher[], ctx.device, :vkCreateGraphicsPipelines)
        res = ccall(fptr, VulkanCore.VkResult,
            (VulkanCore.VkDevice, VulkanCore.VkPipelineCache, UInt32, Ptr{VulkanCore.VkGraphicsPipelineCreateInfo}, Ptr{Cvoid}, Ptr{VulkanCore.VkPipeline}),
            ctx.device.vks, VulkanCore.VkPipelineCache(C_NULL), 1, pointer(ci_arr), C_NULL, pipeline_ref)
        res != VulkanCore.VK_SUCCESS && error("vkCreateGraphicsPipelines failed: $res")
    end

    return Pipeline(pipeline_ref[], ctx.device, Threads.Atomic{UInt64}(1))
end

# ─── UBO update ─────────────────────────────────────────────────────────

"""
    update_ubo!(ctx, state, texture_specs)

Writes per-texture parameters (visibility, min/max, color, etc.) into
the UBO buffer. Skips entirely when `state.ubo_dirty` is false.

**Zero-allocation**: uses pre-allocated `state._ubo_scratch` buffer
and `unsafe_store!` for field packing (no `reinterpret` vectors).

Call `state.ubo_dirty = true` whenever texture specs change (ctrl+scroll,
visibility toggle, maskContribution change, etc.).
"""
function update_ubo!(ctx::VkCtx, state::VkPipelineState, texture_specs)
    # Skip entirely if nothing has changed since last write
    state.ubo_dirty || return
    
    n = min(length(texture_specs), state.n_textures)
    buf = state._ubo_scratch  # Pre-allocated, no zeros() alloc
    fill!(buf, 0x00)          # Clear in-place

    for i in 1:n
        spec = texture_specs[i]
        offset = (i - 1) * TEXTURE_PARAMS_SIZE

        # Pack fields into std140 layout (all via unsafe_store!, 0 allocs)
        _write_int32!(buf, offset + 0, spec.isVisible ? 1 : 0)
        _write_float32!(buf, offset + 4, isempty(spec.minAndMaxValue) ? 0.0f0 : Float32(spec.minAndMaxValue[1]))
        _write_float32!(buf, offset + 8, isempty(spec.minAndMaxValue) ? 1.0f0 : Float32(spec.minAndMaxValue[2]))
        range_val = isempty(spec.minAndMaxValue) ? 1.0f0 : Float32(spec.minAndMaxValue[2] - spec.minAndMaxValue[1])
        _write_float32!(buf, offset + 12, max(range_val, 1.0f0))
        # maskContribution: direct value from 0.0 to 1.0
        mc = clamp(Float32(spec.maskContribution), 0.0f0, 1.0f0)
        _write_float32!(buf, offset + 16, mc)

        # colorMask at offset 32 (16 bytes aligned)
        color = spec.colorMask
        _write_float32!(buf, offset + 32, Float32(color.r))
        _write_float32!(buf, offset + 36, Float32(color.g))
        _write_float32!(buf, offset + 40, Float32(color.b))
        _write_float32!(buf, offset + 44, Float32(color.alpha))

        # allowedIDs at offset 48 + 16 = 64
        n_allowed = isempty(spec.allowedIDs) ? 0 : min(length(spec.allowedIDs), 16)
        _write_int32!(buf, offset + 48, Int32(n_allowed))

        for j in 1:n_allowed
            _write_float32!(buf, offset + 64 + (j - 1) * 16, Float32(spec.allowedIDs[j]))
        end
    end

    # Copy to cached mapped UBO pointer (HOST_COHERENT, no flush needed)
    unsafe_copyto!(state.mapped_ubo_ptr, pointer(buf), state.ubo_size)
    state.ubo_dirty = false
end

@inline function _write_int32!(buf, offset, val::Integer)
    # Direct unsafe_store! — zero allocations (no reinterpret/Vector overhead)
    unsafe_store!(Ptr{Int32}(pointer(buf, offset + 1)), Int32(val))
end

@inline function _write_float32!(buf, offset, val::Real)
    # Direct unsafe_store! — zero allocations (no reinterpret/Vector overhead)
    unsafe_store!(Ptr{Float32}(pointer(buf, offset + 1)), Float32(val))
end

# ─── Descriptor set texture update ──────────────────────────────────────

"""
    update_descriptor_textures!(ctx, state, textures)

Binds an array of `VkTexture` objects to the sampler descriptor set (set 0).
"""
function update_descriptor_textures!(ctx::VkCtx, state::VkPipelineState, textures::Vector{VkTexture})
    writes = WriteDescriptorSet[]

    for (i, tex) in enumerate(textures[1:min(length(textures), state.n_textures)])
        image_info = DescriptorImageInfo(
            tex.sampler,
            tex.view,
            IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
        )
        write = WriteDescriptorSet(
            state.descriptor_set_samplers,  # set 0
            UInt32(i - 1),                  # binding
            0,                              # array element
            DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            [image_info],
            [],
            []
        )
        push!(writes, write)
    end

    !isempty(writes) && update_descriptor_sets(ctx.device, writes, [])
end

# ─── Cleanup ────────────────────────────────────────────────────────────

function destroy_pipeline_state!(ctx::VkCtx, state::VkPipelineState)
    unwrap(device_wait_idle(ctx.device))
end

end # module VulkanPipeline
