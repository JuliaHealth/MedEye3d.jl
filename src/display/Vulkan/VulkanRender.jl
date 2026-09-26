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
using ..VulkanPipeline: VkPipelineState, VkVectorPipelineState
using ..VulkanBuffers: VkQuadBuffers, create_buffer
using ..VulkanTextures: VkTexture

export render_frame!, PanelRenderData, ensure_vector_vbo!, destroy_vector_vbo!

# ─── Per-panel render data ──────────────────────────────────────────────

"""
    PanelRenderData

Contains all the Vulkan state needed to render a single panel within
a frame.
"""
struct PanelRenderData
    pipeline_state::VkPipelineState
    vector_pipeline_state::Union{VkVectorPipelineState, Nothing}
    quad_buffers::Union{VkQuadBuffers, Nothing}
    # Push constant data: uvScale (vec2) + uvOffset (vec2) = 16 bytes
    push_constants::Vector{Float32}  # [scale_x, scale_y, offset_x, offset_y]
    # Viewport region in pixels
    viewport_x::Float32
    viewport_y::Float32
    viewport_w::Float32
    viewport_h::Float32
    # Vector primitives to draw (vec2 pos + vec4 color = 6 floats per vertex)
    vector_vertices::Vector{Float32}
    # Pre-allocated vector VBO: Tuple{Buffer, DeviceMemory, Int} or nothing
    vector_vbo::Any
end

# Default constructor for Zero-VBO panel
PanelRenderData(ps::VkPipelineState, vps::Union{VkVectorPipelineState, Nothing}, push_constants::Vector{Float32}, vx, vy, vw, vh, vector_vertices::Vector{Float32}=Float32[], vector_vbo=nothing) =
    PanelRenderData(ps, vps, nothing, push_constants, Float32(vx), Float32(vy), Float32(vw), Float32(vh), vector_vertices, vector_vbo)
    
PanelRenderData(ps::VkPipelineState, push_constants::Vector{Float32}, vx, vy, vw, vh) =
    PanelRenderData(ps, nothing, nothing, push_constants, Float32(vx), Float32(vy), Float32(vw), Float32(vh), Float32[], nothing)

# ─── Pre-allocated scratch arrays to eliminate per-frame heap allocs ────
# These module-level arrays are reused every frame instead of creating
# new 1-element Vectors on each render_frame! call.

const _fence_buf     = Fence[]          # sized to 1 at first use
const _vp_buf        = Viewport[]       # sized to 1 at first use
const _sc_buf        = Rect2D[]         # sized to 1 per Rect2D
const _ds_buf        = DescriptorSet[]  # sized to 2 (samplers + ubo)
const _EMPTY_UINT32  = UInt32[]         # immutable empty dynamic offsets
const _vb_buf        = Buffer[]         # sized to 1 for vertex buffer bind
const _vb_off_buf    = UInt64[0]        # vertex buffer offset (always 0)
const _wait_sem_buf  = Semaphore[]      # sized to 1
const _sig_sem_buf   = Semaphore[]      # sized to 1
const _cmd_buf       = CommandBuffer[]  # sized to 1
const _clear_buf     = ClearValue[]     # sized to 1
const _submit_buf    = SubmitInfo[]     # sized to 1
const _present_sem   = Semaphore[]      # sized to 1
const _present_sc    = SwapchainKHR[]   # sized to 1
const _present_idx   = UInt32[]         # sized to 1
const _wait_stage_buf = UInt32[VulkanCore.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT]

# Helper to ensure a 1-element buffer is correctly sized and filled
@inline function _set1!(buf::Vector{T}, val::T) where T
    if length(buf) != 1
        resize!(buf, 1)
    end
    @inbounds buf[1] = val
    return buf
end

@inline function _set2!(buf::Vector{T}, a::T, b::T) where T
    if length(buf) != 2
        resize!(buf, 2)
    end
    @inbounds buf[1] = a
    @inbounds buf[2] = b
    return buf
end

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

**Zero per-frame heap allocations**: all wrapper arrays are pre-allocated
module-level buffers that are reused every frame.
"""
function render_frame!(ctx::VkCtx, panels::Vector{PanelRenderData}, target_window=nothing)::Bool
    
    tgt = target_window === nothing ? ctx : target_window

    # Wait for previous frame (reuse fence buffer)
    unwrap(wait_for_fences(ctx.device, _set1!(_fence_buf, tgt.in_flight_fence), true, typemax(UInt64)))

    # Acquire next image
    result = acquire_next_image_khr(ctx.device, tgt.swapchain, typemax(UInt64);
                                     semaphore=tgt.image_available_semaphore)
    local img_idx::UInt32
    try
        img_idx, _ = unwrap(result)
    catch e
        # ERROR_OUT_OF_DATE_KHR or ERROR_SURFACE_LOST_KHR → swapchain needs recreation
        println("Vulkan acquire error (swapchain out of date): $(e)"); flush(stdout)
        return false
    end
    
    # Reset fence ONLY if acquire succeeded, otherwise we deadlock on the next frame
    unwrap(reset_fences(ctx.device, _set1!(_fence_buf, tgt.in_flight_fence)))
    # ctx.last_rendered_image_idx (omitted for secondary)
    if target_window === nothing; ctx.last_rendered_image_idx = img_idx; end

    # Record command buffer
    cmd = tgt.command_buffers[img_idx + 1]  # 0-indexed image, 1-indexed Julia
    unwrap(reset_command_buffer(cmd))

    begin_info = CommandBufferBeginInfo(flags=COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT)
    unwrap(begin_command_buffer(cmd, begin_info))

    # Begin render pass (reuse clear value buffer)
    _set1!(_clear_buf, ClearValue(ClearColorValue((0.0f0, 0.0f0, 0.0f0, 1.0f0))))
    rpbi = RenderPassBeginInfo(
        tgt.render_pass,
        tgt.framebuffers[img_idx + 1],
        Rect2D(Offset2D(0, 0), tgt.swapchain_extent),
        _clear_buf
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
        _set1!(_vp_buf, Viewport(panel.viewport_x, panel.viewport_y + panel.viewport_h,
                       panel.viewport_w, -panel.viewport_h,
                       0.0f0, 1.0f0))
        cmd_set_viewport(cmd, _vp_buf)

        _set1!(_sc_buf, Rect2D(
            Offset2D(round(Int32, panel.viewport_x), round(Int32, panel.viewport_y)),
            Extent2D(round(UInt32, max(panel.viewport_w, 1.0f0)), round(UInt32, max(panel.viewport_h, 1.0f0)))
        ))
        cmd_set_scissor(cmd, _sc_buf)

        # Push constants (zoom/pan)
        pc_data = panel.push_constants
        GC.@preserve pc_data begin
            cmd_push_constants(cmd, ps.pipeline_layout,
                               SHADER_STAGE_VERTEX_BIT | SHADER_STAGE_FRAGMENT_BIT,
                               0, sizeof(pc_data),
                               Ptr{Cvoid}(pointer(pc_data)))
        end

        # Bind descriptor sets (reuse pre-allocated buffers)
        _set2!(_ds_buf, ps.descriptor_set_samplers, ps.descriptor_set_ubo)
        cmd_bind_descriptor_sets(cmd, PIPELINE_BIND_POINT_GRAPHICS,
                                  ps.pipeline_layout,
                                  0,  # first set
                                  _ds_buf,
                                  _EMPTY_UINT32)

        # Bind vertex/index buffers or issue procedural Zero-VBO draw
        if panel.quad_buffers !== nothing
            qb = panel.quad_buffers
            _set1!(_vb_buf, qb.vertex_buffer)
            cmd_bind_vertex_buffers(cmd, _vb_buf, _vb_off_buf)
            cmd_bind_index_buffer(cmd, qb.index_buffer, 0, INDEX_TYPE_UINT32)
            cmd_draw_indexed(cmd, UInt32(qb.index_count), 1, 0, 0, 0)
        else
            # Zero-VBO: 6 procedural vertices (2 triangles) generated via gl_VertexIndex
            cmd_draw(cmd, 6, 1, 0, 0)
        end
        
        # --- VECTOR PIPELINE OVERLAY ---
        if !isempty(panel.vector_vertices) && panel.vector_pipeline_state !== nothing && panel.vector_vbo !== nothing
            vps = panel.vector_pipeline_state
            
            # Map buffer and copy data
            
            
            data_size = sizeof(Float32) * length(panel.vector_vertices)
            vb, mem, _cap = panel.vector_vbo
            
            ptr = unwrap(Vulkan.map_memory(ctx.device, mem, 0, data_size))
            unsafe_copyto!(Ptr{Float32}(ptr), pointer(panel.vector_vertices), length(panel.vector_vertices))
            Vulkan.unmap_memory(ctx.device, mem)
            
            # Bind vector pipeline
            cmd_bind_pipeline(cmd, PIPELINE_BIND_POINT_GRAPHICS, vps.pipeline)
            
            # Re-push constants for the vector pipeline layout (separate VkPipelineLayout handle)
            GC.@preserve pc_data begin
                cmd_push_constants(cmd, vps.pipeline_layout,
                                   SHADER_STAGE_VERTEX_BIT | SHADER_STAGE_FRAGMENT_BIT,
                                   0, sizeof(pc_data),
                                   Ptr{Cvoid}(pointer(pc_data)))
            end
            
            # Bind scratch vertex buffer
            _set1!(_vb_buf, vb)
            cmd_bind_vertex_buffers(cmd, _vb_buf, _vb_off_buf)
            
            # Draw
            n_vertices = length(panel.vector_vertices) ÷ 6
            cmd_draw(cmd, UInt32(n_vertices), 1, 0, 0)
        end

    end

    cmd_end_render_pass(cmd)
    unwrap(end_command_buffer(cmd))

    # Submit (reuse pre-allocated wrapper arrays)
    _set1!(_wait_sem_buf, tgt.image_available_semaphore)
    _set1!(_cmd_buf, cmd)
    _set1!(_sig_sem_buf, tgt.render_finished_semaphore)
    submit_info = SubmitInfo(
        _wait_sem_buf,
        _wait_stage_buf,
        _cmd_buf,
        _sig_sem_buf
    )
    _set1!(_submit_buf, submit_info)
    unwrap(queue_submit(ctx.graphics_queue, _submit_buf; fence=tgt.in_flight_fence))

    # Present (reuse pre-allocated wrapper arrays)
    _set1!(_present_sem, tgt.render_finished_semaphore)
    _set1!(_present_sc, tgt.swapchain)
    _set1!(_present_idx, UInt32(img_idx))
    present_info = PresentInfoKHR(
        _present_sem,
        _present_sc,
        _present_idx
    )
    try
        queue_present_khr(ctx.graphics_queue, present_info)
    catch e
        println("Vulkan present error (swapchain out of date): $(e)"); flush(stdout)
        return false  # swapchain needs recreation
    end

    return true
end

# ─── Vector VBO lifecycle ───────────────────────────────────────────────

"""
    ensure_vector_vbo!(ctx, current, required_size) → Tuple{Buffer, DeviceMemory, Int}

Returns a host-visible vertex buffer large enough for `required_size` bytes.
Re-uses `current` if large enough; otherwise destroys it and allocates a new one
with 2× headroom to reduce re-allocations.
"""
function ensure_vector_vbo!(ctx::VkCtx, current::Any, required_size::Integer)
    if current !== nothing
        buf, mem, cap = current
        if cap >= required_size
            return current
        end
        # Old buffer too small — destroy it
        Vulkan.destroy_buffer(ctx.device, buf)
        Vulkan.free_memory(ctx.device, mem)
    end
    alloc_size = max(required_size, 4096) * 2
    buf, mem = create_buffer(ctx, alloc_size,
        BUFFER_USAGE_VERTEX_BUFFER_BIT,
        MEMORY_PROPERTY_HOST_VISIBLE_BIT | MEMORY_PROPERTY_HOST_COHERENT_BIT)
    return (buf, mem, alloc_size)
end

"""
    destroy_vector_vbo!(ctx, vbo_tuple)

Destroys the vector VBO buffer and frees its device memory.
"""
function destroy_vector_vbo!(ctx::VkCtx, vbo_tuple)
    if vbo_tuple !== nothing
        buf, mem, _ = vbo_tuple
        Vulkan.destroy_buffer(ctx.device, buf)
        Vulkan.free_memory(ctx.device, mem)
    end
end

end # module VulkanRender
