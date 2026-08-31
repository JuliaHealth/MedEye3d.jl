"""
    VulkanTextures

Manages Vulkan texture images: creation, staging buffer uploads, image
layout transitions, image views, and samplers.

Replaces OpenGL `glTexImage2D` / `glTexSubImage2D` / `glActiveTexture` +
`glBindTexture` operations from `TextureManag.jl`.
"""
module VulkanTextures

using Vulkan
using ..VulkanContext: VkCtx, find_memory_type, one_time_submit!
using ..VulkanBuffers: create_buffer

export VkTexture, create_vulkan_texture, update_vulkan_texture!, destroy_vulkan_texture!

# ─── VkTexture: all handles for one texture ─────────────────────────────

"""
    VkTexture

Holds the Vulkan handles for a single 2D texture: image, memory, view,
sampler, and a persistent staging buffer for fast re-uploads.
"""
mutable struct VkTexture
    image::Image
    memory::DeviceMemory
    view::ImageView
    sampler::Sampler
    width::Int
    height::Int
    format::Format
    # Persistent staging buffer for fast texture updates
    staging_buffer::Buffer
    staging_memory::DeviceMemory
    staging_size::Int
    # Name (from TextureSpec.name) for matching
    name::String
end

# ─── Image layout transition helper ─────────────────────────────────────

"""
    transition_image_layout!(cmd, image, old_layout, new_layout)

Records a pipeline barrier to transition an image between layouts.
"""
function transition_image_layout!(cmd::CommandBuffer, image::Image,
                                   old_layout::ImageLayout, new_layout::ImageLayout)
    # Determine access masks and pipeline stages based on transition
    src_access, dst_access, src_stage, dst_stage = if old_layout == IMAGE_LAYOUT_UNDEFINED && new_layout == IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
        (AccessFlag(0), ACCESS_TRANSFER_WRITE_BIT,
         PIPELINE_STAGE_TOP_OF_PIPE_BIT, PIPELINE_STAGE_TRANSFER_BIT)
    elseif old_layout == IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && new_layout == IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
        (ACCESS_TRANSFER_WRITE_BIT, ACCESS_SHADER_READ_BIT,
         PIPELINE_STAGE_TRANSFER_BIT, PIPELINE_STAGE_FRAGMENT_SHADER_BIT)
    elseif old_layout == IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL && new_layout == IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
        (ACCESS_SHADER_READ_BIT, ACCESS_TRANSFER_WRITE_BIT,
         PIPELINE_STAGE_FRAGMENT_SHADER_BIT, PIPELINE_STAGE_TRANSFER_BIT)
    else
        (ACCESS_MEMORY_READ_BIT | ACCESS_MEMORY_WRITE_BIT,
         ACCESS_MEMORY_READ_BIT | ACCESS_MEMORY_WRITE_BIT,
         PIPELINE_STAGE_ALL_COMMANDS_BIT, PIPELINE_STAGE_ALL_COMMANDS_BIT)
    end

    barrier = ImageMemoryBarrier(
        src_access,
        dst_access,
        old_layout,
        new_layout,
        QUEUE_FAMILY_IGNORED,
        QUEUE_FAMILY_IGNORED,
        image,
        ImageSubresourceRange(IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1)
    )
    cmd_pipeline_barrier(cmd, MemoryBarrier[], BufferMemoryBarrier[], [barrier];
                         src_stage_mask=src_stage, dst_stage_mask=dst_stage)
end

# ─── Texture creation ───────────────────────────────────────────────────

"""
    create_vulkan_texture(ctx, width, height, format, data; filter_mode, name) → VkTexture

Creates a Vulkan texture image with the given dimensions and format,
uploads `data` via a staging buffer, and creates an image view and sampler.

`filter_mode`: `:linear` for continuous data (CT, PET), `:nearest` for
discrete masks. Default `:linear`.
"""
function create_vulkan_texture(ctx::VkCtx, width::Int, height::Int,
                                format::Format, data::AbstractArray;
                                filter_mode::Symbol=:linear,
                                name::String="")::VkTexture
    # ── Create image ──
    ici = ImageCreateInfo(
        IMAGE_TYPE_2D,
        format,
        Extent3D(UInt32(width), UInt32(height), 1),
        1, 1,  # mip levels, array layers
        SAMPLE_COUNT_1_BIT,
        IMAGE_TILING_OPTIMAL,
        IMAGE_USAGE_TRANSFER_DST_BIT | IMAGE_USAGE_SAMPLED_BIT,
        SHARING_MODE_EXCLUSIVE,
        UInt32[],
        IMAGE_LAYOUT_UNDEFINED
    )
    image = unwrap(create_image(ctx.device, ici))

    # Allocate and bind memory
    mem_reqs = get_image_memory_requirements(ctx.device, image)
    mem_type = find_memory_type(ctx.physical_device, mem_reqs.memory_type_bits,
                                MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
    mem = unwrap(allocate_memory(ctx.device, MemoryAllocateInfo(mem_reqs.size, mem_type)))
    unwrap(bind_image_memory(ctx.device, image, mem, 0))

    # ── Staging buffer (persistent — reused for texture updates) ──
    data_bytes = collect(reinterpret(UInt8, vec(data)))
    staging_size = max(length(data_bytes), 4)  # at least 4 bytes
    staging_buf, staging_mem = create_buffer(ctx, staging_size,
        BUFFER_USAGE_TRANSFER_SRC_BIT,
        MEMORY_PROPERTY_HOST_VISIBLE_BIT | MEMORY_PROPERTY_HOST_COHERENT_BIT)

    # Upload initial data
    ptr = unwrap(map_memory(ctx.device, staging_mem, 0, UInt64(staging_size)))
    unsafe_copyto!(Ptr{UInt8}(ptr), pointer(data_bytes), length(data_bytes))
    unmap_memory(ctx.device, staging_mem)

    # Transition UNDEFINED → TRANSFER_DST, copy, then TRANSFER_DST → SHADER_READ
    one_time_submit!(ctx) do cmd
        transition_image_layout!(cmd, image, IMAGE_LAYOUT_UNDEFINED, IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)

        region = BufferImageCopy(
            0,  # buffer offset
            0, 0,  # buffer row length, image height (tightly packed)
            ImageSubresourceLayers(IMAGE_ASPECT_COLOR_BIT, 0, 0, 1),
            Offset3D(0, 0, 0),
            Extent3D(UInt32(width), UInt32(height), 1)
        )
        cmd_copy_buffer_to_image(cmd, staging_buf, image, IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, [region])

        transition_image_layout!(cmd, image, IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
    end

    # ── Image view ──
    ivci = ImageViewCreateInfo(
        image,
        IMAGE_VIEW_TYPE_2D,
        format,
        ComponentMapping(
            COMPONENT_SWIZZLE_IDENTITY,
            COMPONENT_SWIZZLE_IDENTITY,
            COMPONENT_SWIZZLE_IDENTITY,
            COMPONENT_SWIZZLE_IDENTITY
        ),
        ImageSubresourceRange(IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1)
    )
    view = unwrap(create_image_view(ctx.device, ivci))

    # ── Sampler ──
    vk_filter = filter_mode == :nearest ? FILTER_NEAREST : FILTER_LINEAR
    sci = SamplerCreateInfo(
        vk_filter, vk_filter,
        SAMPLER_MIPMAP_MODE_NEAREST,
        SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        0.0f0,    # mipLodBias
        false,    # anisotropyEnable
        1.0f0,    # maxAnisotropy
        false,    # compareEnable
        COMPARE_OP_ALWAYS,
        0.0f0,    # minLod
        0.0f0,    # maxLod
        BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
        false     # unnormalizedCoordinates
    )
    sampler = unwrap(create_sampler(ctx.device, sci))

    return VkTexture(image, mem, view, sampler, width, height, format,
                     staging_buf, staging_mem, staging_size, name)
end

# ─── Texture update (sub-image upload) ──────────────────────────────────

"""
    update_vulkan_texture!(ctx, tex, data)

Updates the texture with new pixel data. Uses the persistent staging
buffer for zero-allocation re-uploads (like `glTexSubImage2D`).
"""
function update_vulkan_texture!(ctx::VkCtx, tex::VkTexture, data::AbstractArray)
    data_bytes = collect(reinterpret(UInt8, vec(data)))
    upload_size = length(data_bytes)

    # Grow staging buffer if needed — old handle will be GC'd by Vulkan.jl finalizers
    if upload_size > tex.staging_size
        unwrap(device_wait_idle(ctx.device))
        tex.staging_buffer, tex.staging_memory = create_buffer(ctx, upload_size,
            BUFFER_USAGE_TRANSFER_SRC_BIT,
            MEMORY_PROPERTY_HOST_VISIBLE_BIT | MEMORY_PROPERTY_HOST_COHERENT_BIT)
        tex.staging_size = upload_size
    end

    # Map and copy
    ptr = unwrap(map_memory(ctx.device, tex.staging_memory, 0, UInt64(upload_size)))
    unsafe_copyto!(Ptr{UInt8}(ptr), pointer(data_bytes), upload_size)
    unmap_memory(ctx.device, tex.staging_memory)

    # Transition, copy, transition back
    one_time_submit!(ctx) do cmd
        transition_image_layout!(cmd, tex.image, IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)

        region = BufferImageCopy(
            0, 0, 0,
            ImageSubresourceLayers(IMAGE_ASPECT_COLOR_BIT, 0, 0, 1),
            Offset3D(0, 0, 0),
            Extent3D(UInt32(tex.width), UInt32(tex.height), 1)
        )
        cmd_copy_buffer_to_image(cmd, tex.staging_buffer, tex.image, IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, [region])

        transition_image_layout!(cmd, tex.image, IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
    end
end

# ─── Cleanup ────────────────────────────────────────────────────────────

"""
    destroy_vulkan_texture!(ctx, tex)

Ensures GPU is idle. Vulkan.jl GC finalizers handle actual resource destruction.
"""
function destroy_vulkan_texture!(ctx::VkCtx, tex::VkTexture)
    unwrap(device_wait_idle(ctx.device))
    # Vulkan.jl handles have GC finalizers — no manual destroy to avoid double-free
end

end # module VulkanTextures
