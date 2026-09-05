"""
    VulkanStaging

High-performance staging buffer subsystem for Vulkan.
Implements a persistent host-visible, host-coherent staging ring buffer
and batched multi-texture transfers with fused SIMD data conversion
to minimize CPU overhead and GPU synchronization stalls during interactive
3D volume slicing (60+ FPS).

**Optimizations over naive approach:**
- Persistent 64 MB mapped staging buffer (no map/unmap per frame)
- Pre-allocated command buffer (no alloc/free per frame)
- Fence-based async sync (no queue_wait_idle stalls)
- Pre-allocated barrier/spec arrays (no per-frame heap allocs)
- Fused copy+convert directly into mapped GPU memory (0 heap allocs)
"""
module VulkanStaging

using Vulkan
using VulkanCore
using ..VulkanContext: VkCtx, find_memory_type, unwrap
using ..VulkanBuffers: create_buffer
using ..VulkanTextures: VkTexture, transition_image_layout!

export VkStagingPool, create_staging_pool, destroy_staging_pool!, upload_textures_batched!, TextureUploadBatchItem

"""
    VkStagingPool

Persistent staging buffer pool allocated in host-visible, host-coherent memory.
Mapped permanently to avoid repeated map/unmap and buffer creation overhead.
Includes a dedicated pre-allocated command buffer and fence for async submissions.
"""
mutable struct VkStagingPool
    buffer::Buffer
    memory::DeviceMemory
    capacity::UInt64
    mapped_ptr::Ptr{UInt8}
    offset::UInt64
    cmd::CommandBuffer
    # Fence for async GPU sync (no queue_wait_idle stalls)
    transfer_fence::Fence
    transfer_pending::Bool
    # Pre-allocated scratch arrays to eliminate per-batch heap allocations
    _copy_specs::Vector{Tuple{VkTexture, UInt64, Int, Int}}
    _dst_barriers::Vector{ImageMemoryBarrier}
    _read_barriers::Vector{ImageMemoryBarrier}
    _empty_mem_barriers::Vector{MemoryBarrier}
    _empty_buf_barriers::Vector{BufferMemoryBarrier}
end

"""
    create_staging_pool(ctx, capacity_mb=64) → VkStagingPool

Allocates a persistent staging buffer of `capacity_mb` megabytes.
"""
function create_staging_pool(ctx::VkCtx, capacity_mb::Int=64)::VkStagingPool
    capacity = UInt64(capacity_mb) * 1024 * 1024
    
    buffer, memory = create_buffer(
        ctx, capacity,
        BUFFER_USAGE_TRANSFER_SRC_BIT,
        MEMORY_PROPERTY_HOST_VISIBLE_BIT | MEMORY_PROPERTY_HOST_COHERENT_BIT
    )
    
    ptr = unwrap(map_memory(ctx.device, memory, 0, capacity))
    mapped_ptr = Ptr{UInt8}(ptr)
    
    cbai = CommandBufferAllocateInfo(ctx.command_pool, COMMAND_BUFFER_LEVEL_PRIMARY, 1)
    cmd = unwrap(allocate_command_buffers(ctx.device, cbai))[1]
    
    # Create a signaled fence so first wait_for_fences succeeds immediately
    transfer_fence = unwrap(create_fence(ctx.device, FenceCreateInfo(flags=FENCE_CREATE_SIGNALED_BIT)))
    
    # Pre-allocate scratch arrays (will be resized as needed, never freed)
    copy_specs = Tuple{VkTexture, UInt64, Int, Int}[]
    dst_barriers = ImageMemoryBarrier[]
    read_barriers = ImageMemoryBarrier[]
    empty_mem = MemoryBarrier[]
    empty_buf = BufferMemoryBarrier[]
    
    return VkStagingPool(buffer, memory, capacity, mapped_ptr, 0, cmd,
                         transfer_fence, false,
                         copy_specs, dst_barriers, read_barriers, empty_mem, empty_buf)
end

"""
    destroy_staging_pool!(ctx, pool)

Cleans up persistent mapped memory for the staging pool.
"""
function destroy_staging_pool!(ctx::VkCtx, pool::VkStagingPool)
    # Wait for any pending transfer to complete before destroying
    if pool.transfer_pending
        unwrap(wait_for_fences(ctx.device, [pool.transfer_fence], true, typemax(UInt64)))
        pool.transfer_pending = false
    end
    unwrap(device_wait_idle(ctx.device))
    try
        unmap_memory(ctx.device, pool.memory)
    catch e
        @warn "Error unmapping staging memory: $e"
    end
end

"""
    TextureUploadBatchItem

Specification for a single texture slice upload in a batched transfer.
"""
struct TextureUploadBatchItem
    texture::VkTexture
    data::AbstractArray
end

# ─── Fused CPU Copy & SIMD Conversion ──────────────────────────────────

"""
    copy_data_to_staging_f32!(dest::Ptr{Float32}, data::AbstractArray, w::Int, h::Int)

Fast fused copy and type conversion directly into mapped GPU staging memory.
Zero heap allocations regardless of input array type (Float32, Int8, UInt16, Float16, SubArray).
"""
@inline function copy_data_to_staging_f32!(dest::Ptr{Float32}, data::Matrix{Float32}, w::Int, h::Int)
    n = min(length(data), w * h)
    unsafe_copyto!(dest, pointer(data), n)
end

@inline function copy_data_to_staging_f32!(dest::Ptr{Float32}, data::SubArray{Float32, 2, Array{Float32, 3}}, w::Int, h::Int)
    n = min(length(data), w * h)
    if stride(data, 1) == 1 && stride(data, 2) == size(data, 1)
        unsafe_copyto!(dest, pointer(data), n)
    else
        @inbounds for i in 1:n
            unsafe_store!(dest, data[i], i)
        end
    end
end

@inline function copy_data_to_staging_f32!(dest::Ptr{Float32}, data::AbstractArray, w::Int, h::Int)
    n = min(length(data), w * h)
    @inbounds for i in 1:n
        unsafe_store!(dest, Float32(data[i]), i)
    end
end

# ─── Int16 staging copy (for R16_SINT mask/anatomy textures) ─────────────

@inline function copy_data_to_staging_i16!(dest::Ptr{Int16}, data::Matrix{Int16}, w::Int, h::Int)
    n = min(length(data), w * h)
    unsafe_copyto!(dest, pointer(data), n)
end

@inline function copy_data_to_staging_i16!(dest::Ptr{Int16}, data::SubArray{Int16, 2}, w::Int, h::Int)
    n = min(length(data), w * h)
    if stride(data, 1) == 1 && stride(data, 2) == size(data, 1)
        unsafe_copyto!(dest, pointer(data), n)
    else
        @inbounds for i in 1:n
            unsafe_store!(dest, data[i], i)
        end
    end
end

@inline function copy_data_to_staging_i16!(dest::Ptr{Int16}, data::AbstractArray, w::Int, h::Int)
    n = min(length(data), w * h)
    @inbounds for i in 1:n
        unsafe_store!(dest, Int16(data[i]), i)
    end
end

# ─── Int8 staging copy (for R8_SINT bone overlay textures) ──────────────

@inline function copy_data_to_staging_i8!(dest::Ptr{Int8}, data::Matrix{Int8}, w::Int, h::Int)
    n = min(length(data), w * h)
    unsafe_copyto!(dest, pointer(data), n)
end

@inline function copy_data_to_staging_i8!(dest::Ptr{Int8}, data::SubArray{Int8, 2}, w::Int, h::Int)
    n = min(length(data), w * h)
    if stride(data, 1) == 1 && stride(data, 2) == size(data, 1)
        unsafe_copyto!(dest, pointer(data), n)
    else
        @inbounds for i in 1:n
            unsafe_store!(dest, data[i], i)
        end
    end
end

@inline function copy_data_to_staging_i8!(dest::Ptr{Int8}, data::AbstractArray, w::Int, h::Int)
    n = min(length(data), w * h)
    @inbounds for i in 1:n
        unsafe_store!(dest, Int8(data[i]), i)
    end
end

# ─── Format-aware helpers ───────────────────────────────────────────────

"""
    bytes_per_pixel(fmt::Format) → Int

Returns bytes per pixel for the given Vulkan format.
"""
function bytes_per_pixel(fmt::Format)
    if fmt == Vulkan.FORMAT_R32_SFLOAT
        return 4
    elseif fmt == Vulkan.FORMAT_R16_SINT || fmt == Vulkan.FORMAT_R16_UINT || fmt == Vulkan.FORMAT_R16_SFLOAT
        return 2
    elseif fmt == Vulkan.FORMAT_R8_SINT || fmt == Vulkan.FORMAT_R8_UINT || fmt == Vulkan.FORMAT_R8_UNORM
        return 1
    else
        return 4  # default to 4 bytes (R32)
    end
end

"""
    copy_data_to_staging!(dest_base::Ptr, data, w, h, fmt::Format)

Format-dispatched staging copy. Calls the correct type-specific copy function.
"""
@inline function copy_data_to_staging!(dest_base::Ptr{Nothing}, data::AbstractArray, w::Int, h::Int, fmt::Format)
    if fmt == Vulkan.FORMAT_R16_SINT || fmt == Vulkan.FORMAT_R16_UINT
        copy_data_to_staging_i16!(Ptr{Int16}(dest_base), data, w, h)
    elseif fmt == Vulkan.FORMAT_R8_SINT || fmt == Vulkan.FORMAT_R8_UINT
        copy_data_to_staging_i8!(Ptr{Int8}(dest_base), data, w, h)
    else
        copy_data_to_staging_f32!(Ptr{Float32}(dest_base), data, w, h)
    end
end

"""
    upload_textures_batched!(ctx, staging_pool, batch::Vector{TextureUploadBatchItem})

Performs a high-speed batched transfer of multiple textures in ONE command buffer submission.

**Key optimizations:**
- Fence-based async sync: waits for PREVIOUS batch's fence (not queue_wait_idle)
- Pre-allocated scratch arrays: no heap allocs for barriers, specs, etc.
- Single command buffer submission for all textures in the batch
"""
function upload_textures_batched!(ctx::VkCtx, pool::VkStagingPool, batch::Vector{TextureUploadBatchItem})
    isempty(batch) && return
    
    # Wait for previous transfer to finish before reusing staging buffer
    # This is MUCH cheaper than queue_wait_idle — it only waits for the
    # specific transfer work, not the entire graphics pipeline.
    if pool.transfer_pending
        unwrap(wait_for_fences(ctx.device, [pool.transfer_fence], true, typemax(UInt64)))
        pool.transfer_pending = false
    end
    unwrap(reset_fences(ctx.device, [pool.transfer_fence]))
    
    # Calculate total bytes needed (format-aware: R32=4B, R16=2B, R8=1B per pixel)
    total_bytes_needed = UInt64(0)
    for item in batch
        bpp = bytes_per_pixel(item.texture.format)
        tex_bytes = UInt64(item.texture.width * item.texture.height * bpp)
        total_bytes_needed += (tex_bytes + 15) & ~UInt64(15)
    end
    
    # Reset staging ring buffer offset if wrap-around is needed
    if pool.offset + total_bytes_needed > pool.capacity
        pool.offset = 0
    end
    
    # 1. Fused copy/convert into mapped staging memory and record copy specs
    # Reuse pre-allocated arrays instead of allocating new ones
    copy_specs = pool._copy_specs
    empty!(copy_specs)
    curr_offset = pool.offset
    
    for item in batch
        tex = item.texture
        data = item.data
        w, h = tex.width, tex.height
        bpp = bytes_per_pixel(tex.format)
        data_bytes = UInt64(w * h * bpp)
        
        # Ensure 16-byte alignment for Vulkan buffer offsets
        aligned_offset = (curr_offset + 15) & ~UInt64(15)
        dest_ptr = Ptr{Nothing}(pool.mapped_ptr + aligned_offset)
        
        # Zero the full texture area first, then copy actual data on top.
        # When plane changes (e.g., axial→coronal), the slice may be smaller
        # than the texture. Zeroing prevents stale garbage pixels from
        # Vulkan reading past the valid data in the Extent3D copy region.
        ccall(:memset, Ptr{Nothing}, (Ptr{Nothing}, Cint, Csize_t), dest_ptr, 0, data_bytes)
        # Copy actual data (uses min(length(data), w*h) so safe for smaller slices)
        copy_data_to_staging!(dest_ptr, data, w, h, tex.format)
        
        push!(copy_specs, (tex, aligned_offset, w, h))
        curr_offset = aligned_offset + data_bytes
    end
    pool.offset = curr_offset
    
    # 2. Record single command buffer for all texture uploads
    cmd = pool.cmd
    begin_info = CommandBufferBeginInfo(flags=COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT)
    unwrap(begin_command_buffer(cmd, begin_info))
    
    # Transition all textures to TRANSFER_DST_OPTIMAL (reuse pre-allocated array)
    dst_barriers = pool._dst_barriers
    empty!(dst_barriers)
    for (tex, _, _, _) in copy_specs
        b = ImageMemoryBarrier(
            ACCESS_SHADER_READ_BIT,
            ACCESS_TRANSFER_WRITE_BIT,
            IMAGE_LAYOUT_UNDEFINED,
            IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            QUEUE_FAMILY_IGNORED,
            QUEUE_FAMILY_IGNORED,
            tex.image,
            ImageSubresourceRange(IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1)
        )
        push!(dst_barriers, b)
    end
    cmd_pipeline_barrier(cmd, pool._empty_mem_barriers, pool._empty_buf_barriers, dst_barriers;
                         src_stage_mask=PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         dst_stage_mask=PIPELINE_STAGE_TRANSFER_BIT)
    
    # Copy from staging buffer to each image
    for (tex, offset, w, h) in copy_specs
        region = BufferImageCopy(
            offset, 0, 0,
            ImageSubresourceLayers(IMAGE_ASPECT_COLOR_BIT, 0, 0, 1),
            Offset3D(0, 0, 0),
            Extent3D(UInt32(w), UInt32(h), 1)
        )
        cmd_copy_buffer_to_image(cmd, pool.buffer, tex.image,
                                 IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, [region])
    end
    
    # Transition all textures to SHADER_READ_ONLY_OPTIMAL (reuse pre-allocated array)
    read_barriers = pool._read_barriers
    empty!(read_barriers)
    for (tex, _, _, _) in copy_specs
        b = ImageMemoryBarrier(
            ACCESS_TRANSFER_WRITE_BIT,
            ACCESS_SHADER_READ_BIT,
            IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            QUEUE_FAMILY_IGNORED,
            QUEUE_FAMILY_IGNORED,
            tex.image,
            ImageSubresourceRange(IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1)
        )
        push!(read_barriers, b)
    end
    cmd_pipeline_barrier(cmd, pool._empty_mem_barriers, pool._empty_buf_barriers, read_barriers;
                         src_stage_mask=PIPELINE_STAGE_TRANSFER_BIT,
                         dst_stage_mask=PIPELINE_STAGE_FRAGMENT_SHADER_BIT)
    
    unwrap(end_command_buffer(cmd))
    
    # Single queue submission with fence (no queue_wait_idle!)
    submit_info = SubmitInfo([], UInt32[], [cmd], [])
    unwrap(queue_submit(ctx.graphics_queue, [submit_info]; fence=pool.transfer_fence))
    pool.transfer_pending = true
    
    # CPU is now FREE to continue recording next frame's render commands
    # while GPU processes the texture transfers asynchronously
end

end # module VulkanStaging
