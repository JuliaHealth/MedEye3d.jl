"""
    VulkanContext

Manages the core Vulkan resources: instance, physical device, logical device,
queue, surface, swapchain, render pass, framebuffers, command pool and
synchronization primitives.

This module replaces `PrepareWindowHelpers.initializeWindow` and
`PrepareWindow.displayAll` for the Vulkan backend.
"""
module VulkanContext

using Vulkan
using VulkanCore
using GLFW
using Logging

export VkCtx, init_vulkan_context, destroy_vulkan_context!, recreate_swapchain!
export find_memory_type, one_time_submit!

# ─── VkCtx: aggregate of all core Vulkan state ─────────────────────────

"""
    VkCtx

Holds all Vulkan handles needed by the rendering pipeline.
Mutable so swapchain recreation can update fields in-place.
"""
mutable struct VkCtx
    instance::Instance
    physical_device::PhysicalDevice
    device::Device
    graphics_queue::Queue
    queue_family_index::UInt32
    surface::SurfaceKHR
    swapchain::SwapchainKHR
    swapchain_format::Format
    swapchain_extent::Extent2D
    swapchain_images::Vector{Image}
    swapchain_image_views::Vector{ImageView}
    render_pass::RenderPass
    framebuffers::Vector{Framebuffer}
    command_pool::CommandPool
    command_buffers::Vector{CommandBuffer}
    # Synchronisation (per-frame, only 1 frame in flight for simplicity)
    image_available_semaphore::Semaphore
    render_finished_semaphore::Semaphore
    in_flight_fence::Fence
    # Window handle
    window::GLFW.Window
    width::Int
    height::Int
    # Track last rendered swapchain image for screenshots
    last_rendered_image_idx::Int
end

# ─── Helper: find a memory type with the required properties ────────────

"""
    find_memory_type(pdev, type_filter, properties) → UInt32

Scans physical device memory types and returns the first index whose
`property_flags` contains all requested `properties`.
"""
function find_memory_type(pdev::PhysicalDevice, type_filter::Integer, properties::MemoryPropertyFlag)::UInt32
    mem_props = get_physical_device_memory_properties(pdev)
    for i in 0:(mem_props.memory_type_count - 1)
        mt = mem_props.memory_types[i + 1]  # Julia 1-indexed
        if (type_filter & (1 << i)) != 0 && (mt.property_flags & properties) == properties
            return UInt32(i)
        end
    end
    error("VulkanContext: failed to find suitable memory type")
end

# ─── Helper: one-time command buffer submission ─────────────────────────

"""
    one_time_submit!(ctx, f)

Allocates a temporary command buffer, calls `f(cmd)` to record commands,
submits to the graphics queue, and waits for completion.
"""
function one_time_submit!(f::Function, ctx::VkCtx)
    alloc_info = CommandBufferAllocateInfo(ctx.command_pool, COMMAND_BUFFER_LEVEL_PRIMARY, 1)
    cmds = unwrap(allocate_command_buffers(ctx.device, alloc_info))
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
# Convenience: allow (ctx, f) order for non-do-block calls
one_time_submit!(ctx::VkCtx, f::Function) = one_time_submit!(f, ctx)

# ─── Queue family selection ─────────────────────────────────────────────

function find_graphics_queue_family(pdev::PhysicalDevice, surface::SurfaceKHR)::UInt32
    props = get_physical_device_queue_family_properties(pdev)
    for (i, p) in enumerate(props)
        idx = UInt32(i - 1)
        has_graphics = (p.queue_flags & QUEUE_GRAPHICS_BIT) == QUEUE_GRAPHICS_BIT
        has_present = unwrap(get_physical_device_surface_support_khr(pdev, idx, surface))
        if has_graphics && has_present
            return idx
        end
    end
    error("VulkanContext: no queue family with graphics + present support")
end

# ─── Swapchain creation ─────────────────────────────────────────────────

function create_swapchain(ctx_or_device, pdev, surface, qfi, width, height;
                          old_swapchain::Union{SwapchainKHR, Nothing}=nothing)
    caps = unwrap(get_physical_device_surface_capabilities_khr(pdev, surface))

    # Choose B8G8R8A8_UNORM / SRGB_NONLINEAR if available
    # Note: Vulkan.jl v0.6.30 uses `surface` as keyword arg for this function
    formats = unwrap(get_physical_device_surface_formats_khr(pdev; surface=surface))
    chosen_fmt = formats[1]
    for f in formats
        if f.format == FORMAT_B8G8R8A8_UNORM && f.color_space == COLOR_SPACE_SRGB_NONLINEAR_KHR
            chosen_fmt = f
            break
        end
    end

    # Clamp extent
    extent = Extent2D(
        clamp(UInt32(width),  caps.min_image_extent.width,  caps.max_image_extent.width),
        clamp(UInt32(height), caps.min_image_extent.height, caps.max_image_extent.height)
    )

    image_count = caps.min_image_count + 1
    if caps.max_image_count > 0
        image_count = min(image_count, caps.max_image_count)
    end

    present_mode = PRESENT_MODE_FIFO_KHR  # guaranteed available, vsync

    device = ctx_or_device isa VkCtx ? ctx_or_device.device : ctx_or_device

    sci = SwapchainCreateInfoKHR(
        surface,
        image_count,
        chosen_fmt.format,
        chosen_fmt.color_space,
        extent,
        1,  # imageArrayLayers
        IMAGE_USAGE_COLOR_ATTACHMENT_BIT | IMAGE_USAGE_TRANSFER_SRC_BIT,
        SHARING_MODE_EXCLUSIVE,
        [qfi],
        caps.current_transform,
        COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        present_mode,
        true;  # clipped
        old_swapchain = old_swapchain === nothing ? C_NULL : old_swapchain
    )

    swapchain = unwrap(create_swapchain_khr(device, sci))
    images = unwrap(get_swapchain_images_khr(device, swapchain))

    # Image views
    views = map(images) do img
        ivci = ImageViewCreateInfo(
            img,
            IMAGE_VIEW_TYPE_2D,
            chosen_fmt.format,
            ComponentMapping(
                COMPONENT_SWIZZLE_IDENTITY,
                COMPONENT_SWIZZLE_IDENTITY,
                COMPONENT_SWIZZLE_IDENTITY,
                COMPONENT_SWIZZLE_IDENTITY
            ),
            ImageSubresourceRange(IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1)
        )
        unwrap(create_image_view(device, ivci))
    end

    return swapchain, chosen_fmt.format, extent, images, views
end

# ─── Render pass ────────────────────────────────────────────────────────

function create_render_pass(device::Device, format::Format)::RenderPass
    att = VulkanCore.VkAttachmentDescription(
        UInt32(0),
        VulkanCore.VkFormat(Int(format)),
        VulkanCore.VK_SAMPLE_COUNT_1_BIT,
        VulkanCore.VK_ATTACHMENT_LOAD_OP_CLEAR,
        VulkanCore.VK_ATTACHMENT_STORE_OP_STORE,
        VulkanCore.VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        VulkanCore.VK_ATTACHMENT_STORE_OP_DONT_CARE,
        VulkanCore.VK_IMAGE_LAYOUT_UNDEFINED,
        VulkanCore.VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
    )
    atts = [att]
    color_ref = VulkanCore.VkAttachmentReference(UInt32(0), VulkanCore.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL)
    color_refs = [color_ref]
    subpass = VulkanCore.VkSubpassDescription(
        UInt32(0),
        VulkanCore.VK_PIPELINE_BIND_POINT_GRAPHICS,
        UInt32(0), C_NULL,
        UInt32(1), pointer(color_refs),
        C_NULL, C_NULL,
        UInt32(0), C_NULL
    )
    subpasses = [subpass]
    dep = VulkanCore.VkSubpassDependency(
        VulkanCore.VK_SUBPASS_EXTERNAL,
        UInt32(0),
        VulkanCore.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        VulkanCore.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        UInt32(0),
        VulkanCore.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
        UInt32(0)
    )
    deps = [dep]

    rp_ref = Ref(VulkanCore.VkRenderPass(C_NULL))
    GC.@preserve atts color_refs subpasses deps begin
        rpci = Ref(VulkanCore.VkRenderPassCreateInfo(
            VulkanCore.VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
            C_NULL, UInt32(0),
            UInt32(1), pointer(atts),
            UInt32(1), pointer(subpasses),
            UInt32(1), pointer(deps)
        ))
        fptr = Vulkan.function_pointer(Vulkan.global_dispatcher[], device, :vkCreateRenderPass)
        res = ccall(fptr, VulkanCore.VkResult,
            (VulkanCore.VkDevice, Ptr{VulkanCore.VkRenderPassCreateInfo}, Ptr{Cvoid}, Ptr{VulkanCore.VkRenderPass}),
            device.vks, rpci, C_NULL, rp_ref)
        res != VulkanCore.VK_SUCCESS && error("vkCreateRenderPass failed: $res")
    end
    return RenderPass(rp_ref[], device, Threads.Atomic{UInt64}(1))
end

# ─── Framebuffers ───────────────────────────────────────────────────────

function create_framebuffers(device::Device, render_pass::RenderPass,
                             image_views::Vector{ImageView}, extent::Extent2D)
    return map(image_views) do iv
        fbci = FramebufferCreateInfo(render_pass, [iv], extent.width, extent.height, 1)
        unwrap(create_framebuffer(device, fbci))
    end
end

# ─── init_vulkan_context ────────────────────────────────────────────────

"""
    init_vulkan_context(window, width, height) → VkCtx

Initialises a Vulkan instance, selects a discrete GPU, creates logical device, 
swapchain, render pass, framebuffers, command pool, and synchronisation objects 
using the provided GLFW window.
"""
function init_vulkan_context(window::GLFW.Window, width::Int, height::Int)::VkCtx

    # ── Instance ──
    app_info = ApplicationInfo(v"1.0.0", v"1.0.0", v"1.2.0"; application_name="MedEye3d", engine_name="MedEye3d")
    glfw_exts = GLFW.GetRequiredInstanceExtensions()
    # GLFW may return Vector{String} or Vector{Ptr{Cchar}} depending on version
    exts = String[e isa String ? e : unsafe_string(e) for e in glfw_exts]
    ici = InstanceCreateInfo([], exts; application_info=app_info)
    instance = unwrap(create_instance(ici))

    # ── Surface via GLFW ──
    surface_ref = Ref{VulkanCore.VkSurfaceKHR}(VulkanCore.VkSurfaceKHR(0))
    err = ccall((:glfwCreateWindowSurface, GLFW.libglfw),
                Cint,
                (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{VulkanCore.VkSurfaceKHR}),
                instance.vks, window.handle, C_NULL, surface_ref)
    err != 0 && error("glfwCreateWindowSurface failed: $err")
    surface = SurfaceKHR(Ptr{Cvoid}(surface_ref[]), instance, Threads.Atomic{UInt64}(1))

    # ── Physical device — prefer discrete GPU ──
    pdevs = unwrap(enumerate_physical_devices(instance))
    isempty(pdevs) && error("No Vulkan physical devices found")
    pdev = pdevs[1]
    for pd in pdevs
        props = get_physical_device_properties(pd)
        if props.device_type == PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
            pdev = pd
            break
        end
    end
    @info "Vulkan GPU: $(get_physical_device_properties(pdev).device_name)"

    # ── Queue family ──
    qfi = find_graphics_queue_family(pdev, surface)

    # ── Logical device ──
    queue_ci = DeviceQueueCreateInfo(qfi, [1.0f0])
    dev_ci = DeviceCreateInfo([queue_ci], [], ["VK_KHR_swapchain"])
    device = unwrap(create_device(pdev, dev_ci))
    queue = get_device_queue(device, qfi, 0)

    # ── Swapchain ──
    swapchain, sc_format, sc_extent, sc_images, sc_views =
        create_swapchain(device, pdev, surface, qfi, width, height)

    # ── Render pass ──
    render_pass = create_render_pass(device, sc_format)

    # ── Framebuffers ──
    framebuffers = create_framebuffers(device, render_pass, sc_views, sc_extent)

    # ── Command pool & buffers ──
    cpci = CommandPoolCreateInfo(qfi; flags=COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT)
    cmd_pool = unwrap(create_command_pool(device, cpci))
    cbai = CommandBufferAllocateInfo(cmd_pool, COMMAND_BUFFER_LEVEL_PRIMARY, UInt32(length(sc_images)))
    cmd_bufs = unwrap(allocate_command_buffers(device, cbai))

    # ── Synchronisation ──
    img_avail  = unwrap(create_semaphore(device, SemaphoreCreateInfo()))
    render_fin = unwrap(create_semaphore(device, SemaphoreCreateInfo()))
    fence      = unwrap(create_fence(device, FenceCreateInfo(flags=FENCE_CREATE_SIGNALED_BIT)))

    return VkCtx(
        instance, pdev, device, queue, qfi,
        surface, swapchain, sc_format, sc_extent, sc_images, sc_views,
        render_pass, framebuffers, cmd_pool, cmd_bufs,
        img_avail, render_fin, fence,
        window, width, height,
        0  # last_rendered_image_idx
    )
end

# ─── Swapchain recreation (window resize) ───────────────────────────────

"""
    recreate_swapchain!(ctx, new_width, new_height)

Destroys old swapchain resources and creates new ones matching the new
window dimensions. Called on window resize.
"""
function recreate_swapchain!(ctx::VkCtx, new_width::Int, new_height::Int)
    unwrap(device_wait_idle(ctx.device))

    # Old framebuffers, image views, and swapchain will be GC'd by Vulkan.jl finalizers
    old_swapchain = ctx.swapchain
    swapchain, sc_format, sc_extent, sc_images, sc_views =
        create_swapchain(ctx, ctx.physical_device, ctx.surface, ctx.queue_family_index,
                         new_width, new_height; old_swapchain=old_swapchain)

    ctx.swapchain = swapchain
    ctx.swapchain_format = sc_format
    ctx.swapchain_extent = sc_extent
    ctx.swapchain_images = sc_images
    ctx.swapchain_image_views = sc_views
    ctx.framebuffers = create_framebuffers(ctx.device, ctx.render_pass, sc_views, sc_extent)
    ctx.width = new_width
    ctx.height = new_height

    # Re-allocate command buffers
    cbai = CommandBufferAllocateInfo(ctx.command_pool, COMMAND_BUFFER_LEVEL_PRIMARY, UInt32(length(sc_images)))
    ctx.command_buffers = unwrap(allocate_command_buffers(ctx.device, cbai))
end

# ─── Cleanup ────────────────────────────────────────────────────────────

"""
    destroy_vulkan_context!(ctx)

Tears down all Vulkan resources and closes the GLFW window.
Vulkan.jl GC finalizers handle most resource destruction.
"""
function destroy_vulkan_context!(ctx::VkCtx)
    unwrap(device_wait_idle(ctx.device))

    # Close the GLFW window (not a Vulkan handle, needs explicit cleanup)
    if ctx.window.handle != C_NULL
        GLFW.DestroyWindow(ctx.window)
        ctx.window.handle = C_NULL
    end

    # All Vulkan handles (device, instance, semaphores, fences, etc.)
    # will be cleaned up by Vulkan.jl's GC finalizers in proper dependency order.
end

end # module VulkanContext
