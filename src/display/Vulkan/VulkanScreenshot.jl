"""
    VulkanScreenshot

Captures the current swapchain image to a PNG file by copying it to a
host-readable staging buffer and decoding the B8G8R8A8 pixels.

Replaces the OpenGL screenshot path using `glReadPixels`.
"""
module VulkanScreenshot

using Vulkan
using VulkanCore
using ..VulkanContext: VkCtx, find_memory_type, one_time_submit!
using ..VulkanBuffers: create_buffer
using Logging
using ColorTypes
using FixedPointNumbers

export capture_screenshot

"""
    capture_screenshot(ctx, output_path::String; image_index::Int=0)

Captures the swapchain image at `image_index` to a PNG file at
`output_path`. The image is transitioned to TRANSFER_SRC, copied to a
host-visible staging buffer, and then decoded from B8G8R8A8 format.

Returns `true` on success.
"""
function capture_screenshot(ctx::VkCtx, output_path::String; image_index::Int=-1)::Bool
    raw_bytes, w, h = _readback_swapchain_pixels(ctx; image_index=image_index)
    _save_bgra_as_ppm(output_path, raw_bytes, w, h)
    return true
end

"""
    capture_screenshot(ctx) → Matrix{RGBA{N0f8}}

Captures the current swapchain image and returns it as a ColorTypes RGBA matrix.
Used by ScreenshotEvent dispatch in the consumer loop.
"""
function capture_screenshot(ctx::VkCtx; image_index::Int=-1)
    raw_bytes, w, h = _readback_swapchain_pixels(ctx; image_index=image_index)
    img = Matrix{RGBA{N0f8}}(undef, h, w)
    for y in 1:h
        for x in 1:w
            idx = ((y - 1) * w + (x - 1)) * 4
            b = raw_bytes[idx + 1]
            g = raw_bytes[idx + 2]
            r = raw_bytes[idx + 3]
            a = raw_bytes[idx + 4]
            img[y, x] = RGBA{N0f8}(reinterpret(N0f8, r), reinterpret(N0f8, g), reinterpret(N0f8, b), reinterpret(N0f8, a))
        end
    end
    return img
end

"""
    _readback_swapchain_pixels(ctx; image_index) → (bytes, w, h)

Internal: reads back raw BGRA pixel bytes from the specified swapchain image.
"""
function _readback_swapchain_pixels(ctx::VkCtx; image_index::Int=-1)
    if image_index < 0
        image_index = ctx.last_rendered_image_idx
    end
    
    unwrap(Vulkan.device_wait_idle(ctx.device))
    
    w = Int(ctx.swapchain_extent.width)
    h = Int(ctx.swapchain_extent.height)
    pixel_size = 4
    buf_size = w * h * pixel_size

    staging, staging_mem = create_buffer(ctx, buf_size,
        BUFFER_USAGE_TRANSFER_DST_BIT,
        MEMORY_PROPERTY_HOST_VISIBLE_BIT | MEMORY_PROPERTY_HOST_COHERENT_BIT)

    src_image = ctx.swapchain_images[image_index + 1]

    one_time_submit!(ctx) do cmd
        barrier_to_src = ImageMemoryBarrier(
            ACCESS_MEMORY_READ_BIT,
            ACCESS_TRANSFER_READ_BIT,
            IMAGE_LAYOUT_PRESENT_SRC_KHR,
            IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            QUEUE_FAMILY_IGNORED, QUEUE_FAMILY_IGNORED,
            src_image,
            ImageSubresourceRange(IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1)
        )
        cmd_pipeline_barrier(cmd, MemoryBarrier[], BufferMemoryBarrier[], [barrier_to_src];
            src_stage_mask=PIPELINE_STAGE_TRANSFER_BIT, dst_stage_mask=PIPELINE_STAGE_TRANSFER_BIT)

        region = BufferImageCopy(
            0, 0, 0,
            ImageSubresourceLayers(IMAGE_ASPECT_COLOR_BIT, 0, 0, 1),
            Offset3D(0, 0, 0),
            Extent3D(UInt32(w), UInt32(h), 1)
        )
        cmd_copy_image_to_buffer(cmd, src_image, IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                  staging, [region])

        barrier_back = ImageMemoryBarrier(
            ACCESS_TRANSFER_READ_BIT,
            ACCESS_MEMORY_READ_BIT,
            IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            IMAGE_LAYOUT_PRESENT_SRC_KHR,
            QUEUE_FAMILY_IGNORED, QUEUE_FAMILY_IGNORED,
            src_image,
            ImageSubresourceRange(IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1)
        )
        cmd_pipeline_barrier(cmd, MemoryBarrier[], BufferMemoryBarrier[], [barrier_back];
            src_stage_mask=PIPELINE_STAGE_TRANSFER_BIT, dst_stage_mask=PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT)
    end

    ptr = unwrap(map_memory(ctx.device, staging_mem, 0, UInt64(buf_size)))
    raw_bytes = Vector{UInt8}(undef, buf_size)
    unsafe_copyto!(pointer(raw_bytes), Ptr{UInt8}(ptr), buf_size)
    unmap_memory(ctx.device, staging_mem)

    return raw_bytes, w, h
end

"""
    _save_bgra_as_ppm(path, bytes, w, h)

Saves B8G8R8A8 raw pixels as a PPM image (portable, no dependencies).
If the path ends in .png, replaces extension with .ppm.
"""
function _save_bgra_as_ppm(path::String, bytes::Vector{UInt8}, w::Int, h::Int)
    # Ensure .ppm extension
    out_path = replace(path, r"\.[^.]*$" => ".ppm")
    if !endswith(out_path, ".ppm")
        out_path *= ".ppm"
    end

    open(out_path, "w") do f
        println(f, "P6")
        println(f, "$w $h")
        println(f, "255")
        for y in 1:h
            for x in 1:w
                idx = ((y - 1) * w + (x - 1)) * 4
                b = bytes[idx + 1]
                g = bytes[idx + 2]
                r = bytes[idx + 3]
                # Write RGB (skip alpha)
                write(f, r)
                write(f, g)
                write(f, b)
            end
        end
    end
    @info "Screenshot saved: $out_path ($(w)x$(h))"
end

end # module VulkanScreenshot
