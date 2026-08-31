"""
    VulkanRender

Frame rendering and presentation for the Vulkan backend.
Records command buffers, submits draw calls, and presents to the swapchain.

Replaces the OpenGL render loop (glClear, glDrawElements, SwapBuffers)
from `SegmentationDisplay.jl`.
"""
module VulkanRender

using Vulkan
using VulkanCore
using ..VulkanContext: VkCtx
using ..VulkanPipeline: VkPipelineState
using ..VulkanBuffers: VkQuadBuffers
using ..VulkanTextures: VkTexture

export render_frame!, PanelRenderData

# ─── Per-panel render data ──────────────────────────────────────────────

"""
    PanelRenderData

Contains all the Vulkan state needed to render a single panel within
a frame.
"""
struct PanelRenderData
    pipeline_state::VkPipelineState
    quad_buffers::Union{VkQuadBuffers, Nothing}
    # Push constant data: uvScale (vec2) + uvOffset (vec2) = 16 bytes
    push_constants::Vector{Float32}  # [scale_x, scale_y, offset_x, offset_y]
    # Viewport region in pixels
    viewport_x::Float32
    viewport_y::Float32
    viewport_w::Float32
    viewport_h::Float32
end

# Default constructor for Zero-VBO panel
PanelRenderData(ps::VkPipelineState, push_constants::Vector{Float32}, vx, vy, vw, vh) =
    PanelRenderData(ps, nothing, push_constants, Float32(vx), Float32(vy), Float32(vw), Float32(vh))

# ─── Frame rendering ───────────────────────────────────────────────────

"""
    render_frame!(ctx, panels::Vector{PanelRenderData})

Renders one complete frame:
1. Wait for previous frame's fence
2. Acquire next swapchain image
3. Record command buffer with render pass, viewport, draw calls per panel
4. Submit to graphics queue
5. Present to swapchain

Returns `true` if rendering succeeded, `false` if swapchain is out of date.
"""
function render_frame!(ctx::VkCtx, panels::Vector{PanelRenderData})::Bool
    # Wait for previous frame
    unwrap(wait_for_fences(ctx.device, [ctx.in_flight_fence], true, typemax(UInt64)))
    unwrap(reset_fences(ctx.device, [ctx.in_flight_fence]))

    # Acquire next image
    result = acquire_next_image_khr(ctx.device, ctx.swapchain, typemax(UInt64);
                                     semaphore=ctx.image_available_semaphore)
    if isa(result, Vulkan.VulkanError)
        # Swapchain out of date → needs recreation
        return false
    end
    img_idx, _ = unwrap(result)
    ctx.last_rendered_image_idx = img_idx

    # Record command buffer
    cmd = ctx.command_buffers[img_idx + 1]  # 0-indexed image, 1-indexed Julia
    unwrap(reset_command_buffer(cmd))

    begin_info = CommandBufferBeginInfo(flags=COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT)
    unwrap(begin_command_buffer(cmd, begin_info))

    # Begin render pass
    clear_val = ClearValue(ClearColorValue((0.0f0, 0.0f0, 0.0f0, 1.0f0)))
    rpbi = RenderPassBeginInfo(
        ctx.render_pass,
        ctx.framebuffers[img_idx + 1],
        Rect2D(Offset2D(0, 0), ctx.swapchain_extent),
        [clear_val]
    )
    cmd_begin_render_pass(cmd, rpbi, SUBPASS_CONTENTS_INLINE)

    # Render each panel
    for panel in panels
        ps = panel.pipeline_state

        # Bind pipeline
        cmd_bind_pipeline(cmd, PIPELINE_BIND_POINT_GRAPHICS, ps.pipeline)

        # Set dynamic viewport and scissor for this panel
        # Use negative height to flip Y-axis, matching OpenGL's Y-up convention.
        # This ensures all existing mouse position calculations (getNewY,
        # getTextureCoordinatesFromScreen) which assume OpenGL coordinates
        # continue to work correctly. (VK_KHR_maintenance1 / Vulkan 1.1)
        vp = Viewport(panel.viewport_x, panel.viewport_y + panel.viewport_h,
                       panel.viewport_w, -panel.viewport_h,
                       0.0f0, 1.0f0)
        cmd_set_viewport(cmd, [vp])

        sc = Rect2D(
            Offset2D(round(Int32, panel.viewport_x), round(Int32, panel.viewport_y)),
            Extent2D(round(UInt32, max(panel.viewport_w, 1.0f0)), round(UInt32, max(panel.viewport_h, 1.0f0)))
        )
        cmd_set_scissor(cmd, [sc])

        # Push constants (zoom/pan)
        pc_data = panel.push_constants
        GC.@preserve pc_data begin
            cmd_push_constants(cmd, ps.pipeline_layout,
                               SHADER_STAGE_VERTEX_BIT,
                               0, sizeof(pc_data),
                               Ptr{Cvoid}(pointer(pc_data)))
        end

        # Bind descriptor sets
        cmd_bind_descriptor_sets(cmd, PIPELINE_BIND_POINT_GRAPHICS,
                                  ps.pipeline_layout,
                                  0,  # first set
                                  [ps.descriptor_set_samplers, ps.descriptor_set_ubo],
                                  UInt32[])

        # Bind vertex/index buffers or issue procedural Zero-VBO draw
        if panel.quad_buffers !== nothing
            qb = panel.quad_buffers
            cmd_bind_vertex_buffers(cmd, [qb.vertex_buffer], UInt64[0])
            cmd_bind_index_buffer(cmd, qb.index_buffer, 0, INDEX_TYPE_UINT32)
            cmd_draw_indexed(cmd, UInt32(qb.index_count), 1, 0, 0, 0)
        else
            # Zero-VBO: 6 procedural vertices (2 triangles) generated via gl_VertexIndex
            cmd_draw(cmd, 6, 1, 0, 0)
        end
    end

    cmd_end_render_pass(cmd)
    unwrap(end_command_buffer(cmd))

    # Submit
    wait_stages = UInt32[VulkanCore.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT]
    submit_info = SubmitInfo(
        [ctx.image_available_semaphore],
        wait_stages,
        [cmd],
        [ctx.render_finished_semaphore]
    )
    unwrap(queue_submit(ctx.graphics_queue, [submit_info]; fence=ctx.in_flight_fence))

    # Present
    present_info = PresentInfoKHR(
        [ctx.render_finished_semaphore],
        [ctx.swapchain],
        [img_idx]
    )
    present_result = queue_present_khr(ctx.graphics_queue, present_info)
    if isa(present_result, Vulkan.VulkanError)
        return false  # swapchain needs recreation
    end

    return true
end

end # module VulkanRender
