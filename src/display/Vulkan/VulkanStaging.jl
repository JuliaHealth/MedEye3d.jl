"""
    VulkanStaging

High-performance staging buffer subsystem for Vulkan.
Implements a persistent host-visible, host-coherent staging ring buffer
and batched multi-texture transfers to minimize driver overhead during
interactive 3D volume slicing (60+ FPS).
"""
module VulkanStaging

using Vulkan
using VulkanCore
using ..VulkanContext: VkCtx, find_memory_type, unwrap
using ..VulkanBuffers: create_buffer
using ..VulkanTextures: VkTexture, transition_image_layout!

export VkStagingPool, create_staging_pool, upload_textures_batched!, TextureUploadBatchItem

"""
    VkStagingPool

Persistent staging buffer pool allocated in host-visible, host-coherent memory.
Mapped permanently to avoid repeated map/unmap and buffer creation overhead.
"""
mutable struct VkStagingPool
    buffer::Buffer
    memory::DeviceMemory
    capacity::UInt64
    mapped_ptr::Ptr{UInt8}
    offset::UInt64
end

"""
    create_staging_pool(ctx, capacity_mb=32) → VkStagingPool

Allocates a persistent staging buffer of `capacity_mb` megabytes.
"""
function create_staging_pool(ctx::VkCtx, capacity_mb::Int=32)::VkStagingPool
    capacity = UInt64(capacity_mb) * 1024 * 1024
    
    buffer, memory = create_buffer(
        ctx, capacity,
        BUFFER_USAGE_TRANSFER_SRC_BIT,
        MEMORY_PROPERTY_HOST_VISIBLE_BIT | MEMORY_PROPERTY_HOST_COHERENT_BIT
    )
    
    ptr = unwrap(map_memory(ctx.device, memory, 0, capacity))
    mapped_ptr = Ptr{UInt8}(ptr)
    
    return VkStagingPool(buffer, memory, capacity, mapped_ptr, 0)
end

"""
    TextureUploadBatchItem

Specification for a single texture slice upload in a batched transfer.
"""
struct TextureUploadBatchItem
    texture::VkTexture
    data::AbstractArray
end

"""
    upload_textures_batched!(ctx, staging_pool, batch::Vector{TextureUploadBatchItem})

Performs a high-speed batched transfer of multiple textures in ONE command buffer submission.
1. Copies all 2D slices into contiguous offsets in the persistent staging pool.
2. Records pipeline barriers: UNDEFINED/SHADER_READ_ONLY → TRANSFER_DST_OPTIMAL.
3. Records `cmd_copy_buffer_to_image` for each texture from its respective staging offset.
4. Records pipeline barriers: TRANSFER_DST_OPTIMAL → SHADER_READ_ONLY_OPTIMAL.
5. Submits once to the graphics queue with a single fence wait.
"""
function upload_textures_batched!(ctx::VkCtx, pool::VkStagingPool, batch::Vector{TextureUploadBatchItem})
    isempty(batch) && return
    
    # Reset staging offset if wrap-around is needed
    total_bytes_needed = UInt64(0)
    for item in batch
        total_bytes_needed += UInt64(sizeof(item.data))
    end
    
    if pool.offset + total_bytes_needed > pool.capacity
        pool.offset = 0
    end
    
    # 1. Copy data into staging memory and compute copy regions
    copy_specs = Tuple{VkTexture, UInt64, Int, Int}[]
    curr_offset = pool.offset
    
    for item in batch
        tex = item.texture
        data = item.data
        data_bytes = sizeof(data)
        
        # Ensure 16-byte alignment
        aligned_offset = (curr_offset + 15) & ~UInt64(15)
        
        dest_ptr = pool.mapped_ptr + aligned_offset
        is_contig = stride(data, 1) == 1 && stride(data, 2) == size(data, 1)
        if is_contig
            unsafe_copyto!(dest_ptr, Ptr{UInt8}(pointer(data)), data_bytes)
        else
            collected = collect(data)
            unsafe_copyto!(dest_ptr, Ptr{UInt8}(pointer(collected)), data_bytes)
        end
        
        push!(copy_specs, (tex, aligned_offset, tex.width, tex.height))
        curr_offset = aligned_offset + UInt64(data_bytes)
    end
    pool.offset = curr_offset
    
    # 2. Record single command buffer for all texture uploads
    cbai = CommandBufferAllocateInfo(ctx.command_pool, COMMAND_BUFFER_LEVEL_PRIMARY, 1)
    cmd = unwrap(allocate_command_buffers(ctx.device, cbai))[1]
    
    begin_info = CommandBufferBeginInfo(flags=COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT)
    unwrap(begin_command_buffer(cmd, begin_info))
    
    # Transition all textures to TRANSFER_DST_OPTIMAL
    dst_barriers = ImageMemoryBarrier[]
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
    cmd_pipeline_barrier(cmd, MemoryBarrier[], BufferMemoryBarrier[], dst_barriers;
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
    
    # Transition all textures to SHADER_READ_ONLY_OPTIMAL
    read_barriers = ImageMemoryBarrier[]
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
    cmd_pipeline_barrier(cmd, MemoryBarrier[], BufferMemoryBarrier[], read_barriers;
                         src_stage_mask=PIPELINE_STAGE_TRANSFER_BIT,
                         dst_stage_mask=PIPELINE_STAGE_FRAGMENT_SHADER_BIT)
    
    unwrap(end_command_buffer(cmd))
    
    # Submit and wait using queue_wait_idle (avoids fence GC finalizer issues)
    submit_info = SubmitInfo([], UInt32[], [cmd], [])
    unwrap(queue_submit(ctx.graphics_queue, [submit_info]))
    unwrap(queue_wait_idle(ctx.graphics_queue))
    
    # Cleanup one-time command buffer
    free_command_buffers(ctx.device, ctx.command_pool, [cmd])
end

end # module VulkanStaging
