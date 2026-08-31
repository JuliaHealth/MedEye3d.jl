"""
    VulkanBuffers

Manages Vulkan vertex and index buffers for quad geometry rendering.
Replaces OpenGL VBO/EBO creation from `PrepareWindowHelpers.jl`.
"""
module VulkanBuffers

using Vulkan
using ..VulkanContext: VkCtx, find_memory_type, one_time_submit!

export VkQuadBuffers, create_quad_buffers, destroy_quad_buffers!
export update_quad_vertices!, create_buffer

# ─── Quad geometry ──────────────────────────────────────────────────────

"""
    VkQuadBuffers

Holds vertex and index buffers plus their device memory for a set of quads.
"""
mutable struct VkQuadBuffers
    vertex_buffer::Buffer
    vertex_memory::DeviceMemory
    index_buffer::Buffer
    index_memory::DeviceMemory
    vertex_count::Int
    index_count::Int
end

# ─── Generic buffer creation helper ────────────────────────────────────

"""
    create_buffer(ctx, size, usage, mem_properties) → (Buffer, DeviceMemory)

Creates a Vulkan buffer with the specified size, usage flags, and memory
properties. Returns the buffer handle and its device memory.
"""
function create_buffer(ctx::VkCtx, size::Integer,
                       usage::BufferUsageFlag,
                       mem_props::MemoryPropertyFlag)
    bci = BufferCreateInfo(UInt64(size), usage, SHARING_MODE_EXCLUSIVE, UInt32[])
    buffer = unwrap(Vulkan.create_buffer(ctx.device, bci))

    mem_reqs = get_buffer_memory_requirements(ctx.device, buffer)
    mem_type = find_memory_type(ctx.physical_device, mem_reqs.memory_type_bits, mem_props)
    mai = MemoryAllocateInfo(mem_reqs.size, mem_type)
    memory = unwrap(allocate_memory(ctx.device, mai))
    unwrap(bind_buffer_memory(ctx.device, buffer, memory, 0))

    return buffer, memory
end

# ─── Quad buffer creation ───────────────────────────────────────────────

"""
    create_quad_buffers(ctx, vertices::Vector{Float32}) → VkQuadBuffers

Creates device-local vertex and index buffers for quad rendering.
Uses staging buffers for the upload.

The vertex layout matches OpenGL: position(3) + color(3) + texcoord(2) = 8 floats per vertex.
Index format: two triangles forming a quad = [0,1,2, 2,3,0].
"""
function create_quad_buffers(ctx::VkCtx, vertices::Vector{Float32})::VkQuadBuffers
    indices = UInt32[0, 1, 2, 2, 3, 0]

    # ── Vertex buffer ──
    vb_size = sizeof(vertices)
    staging_vb, staging_vb_mem = create_buffer(ctx, vb_size,
        BUFFER_USAGE_TRANSFER_SRC_BIT,
        MEMORY_PROPERTY_HOST_VISIBLE_BIT | MEMORY_PROPERTY_HOST_COHERENT_BIT)

    # Map, copy, unmap
    ptr = unwrap(map_memory(ctx.device, staging_vb_mem, 0, UInt64(vb_size)))
    unsafe_copyto!(Ptr{Float32}(ptr), pointer(vertices), length(vertices))
    unmap_memory(ctx.device, staging_vb_mem)

    vertex_buffer, vertex_memory = create_buffer(ctx, vb_size,
        BUFFER_USAGE_TRANSFER_DST_BIT | BUFFER_USAGE_VERTEX_BUFFER_BIT,
        MEMORY_PROPERTY_DEVICE_LOCAL_BIT)

    # Copy staging → device-local
    one_time_submit!(ctx) do cmd
        region = BufferCopy(0, 0, UInt64(vb_size))
        cmd_copy_buffer(cmd, staging_vb, vertex_buffer, [region])
    end

    # Let Vulkan.jl's GC finalizers clean up staging buffers
    # (manual destroy_buffer/free_memory causes double-free with GC)
    staging_vb = nothing
    staging_vb_mem = nothing

    # ── Index buffer ──
    ib_size = sizeof(indices)
    staging_ib, staging_ib_mem = create_buffer(ctx, ib_size,
        BUFFER_USAGE_TRANSFER_SRC_BIT,
        MEMORY_PROPERTY_HOST_VISIBLE_BIT | MEMORY_PROPERTY_HOST_COHERENT_BIT)

    ptr = unwrap(map_memory(ctx.device, staging_ib_mem, 0, UInt64(ib_size)))
    unsafe_copyto!(Ptr{UInt32}(ptr), pointer(indices), length(indices))
    unmap_memory(ctx.device, staging_ib_mem)

    index_buffer, index_memory = create_buffer(ctx, ib_size,
        BUFFER_USAGE_TRANSFER_DST_BIT | BUFFER_USAGE_INDEX_BUFFER_BIT,
        MEMORY_PROPERTY_DEVICE_LOCAL_BIT)

    one_time_submit!(ctx) do cmd
        region = BufferCopy(0, 0, UInt64(ib_size))
        cmd_copy_buffer(cmd, staging_ib, index_buffer, [region])
    end

    staging_ib = nothing
    staging_ib_mem = nothing

    return VkQuadBuffers(vertex_buffer, vertex_memory, index_buffer, index_memory,
                         div(length(vertices), 8),  # 8 floats per vertex
                         length(indices))
end

"""
    update_quad_vertices!(ctx, quad_bufs, new_vertices)

Re-uploads vertex data to the quad's vertex buffer via a staging buffer.
Used when the panel layout changes (e.g., switching between single/quad view).
"""
function update_quad_vertices!(ctx::VkCtx, quad_bufs::VkQuadBuffers, new_vertices::Vector{Float32})
    vb_size = sizeof(new_vertices)
    staging, staging_mem = create_buffer(ctx, vb_size,
        BUFFER_USAGE_TRANSFER_SRC_BIT,
        MEMORY_PROPERTY_HOST_VISIBLE_BIT | MEMORY_PROPERTY_HOST_COHERENT_BIT)

    ptr = unwrap(map_memory(ctx.device, staging_mem, 0, UInt64(vb_size)))
    unsafe_copyto!(Ptr{Float32}(ptr), pointer(new_vertices), length(new_vertices))
    unmap_memory(ctx.device, staging_mem)

    one_time_submit!(ctx) do cmd
        region = BufferCopy(0, 0, UInt64(vb_size))
        cmd_copy_buffer(cmd, staging, quad_bufs.vertex_buffer, [region])
    end

    # Let GC handle staging cleanup
    quad_bufs.vertex_count = div(length(new_vertices), 8)
end

"""
    destroy_quad_buffers!(ctx, quad_bufs)

Signals GPU idle then lets Vulkan.jl GC finalizers clean up buffer handles.
"""
function destroy_quad_buffers!(ctx::VkCtx, quad_bufs::VkQuadBuffers)
    # Ensure GPU is done with these buffers
    unwrap(device_wait_idle(ctx.device))
    # Vulkan.jl handles have GC finalizers that call vkDestroyBuffer/vkFreeMemory.
    # We don't manually destroy to avoid double-free.
end

end # module VulkanBuffers
