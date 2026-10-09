import re

with open('src/display/Vulkan/VulkanHeatDiffusion.jl', 'r') as f:
    content = f.read()

# Replace seed_only!
old_seed = """function seed_only!(state::HeatDiffusionState, ctx, seed_x::Int, seed_y::Int, seed_z::Int)"""
new_seed = """function seed_only!(state::HeatDiffusionState, ctx, seed_x::Int, seed_y::Int, seed_z::Int; seed_radius::Int=3)"""
content = content.replace(old_seed, new_seed)

old_seed_dispatch = """    # Seed init into ping buffer
    cmd_bind_pipeline(cmd, PIPELINE_BIND_POINT_COMPUTE, state.pipeline_seed_init)
    cmd_bind_descriptor_sets(cmd, PIPELINE_BIND_POINT_COMPUTE, state.layout_seed_init, 0, [state.ds_seed_ping], [])
    # NB: Julia is 1-based but GLSL gl_GlobalInvocationID is 0-based → subtract 1.
    seed_data = Int32[seed_x - 1, seed_y - 1, seed_z - 1, 0]
    GC.@preserve seed_data begin
        cmd_push_constants(cmd, state.layout_seed_init, SHADER_STAGE_COMPUTE_BIT, 0, sizeof(seed_data), Ptr{Cvoid}(pointer(seed_data)))
    end
    cmd_dispatch(cmd, gx, gy, gz)"""

new_seed_dispatch = """    # Clear both ping and pong buffers to 0.0 (fast fixed-function clear)
    clear_color = ClearColorValue(0.0f0, 0.0f0, 0.0f0, 0.0f0)
    subres = ImageSubresourceRange(IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1)
    cmd_clear_color_image(cmd, state.img_u_ping, IMAGE_LAYOUT_GENERAL, clear_color, [subres])
    cmd_clear_color_image(cmd, state.img_u_pong, IMAGE_LAYOUT_GENERAL, clear_color, [subres])
    
    # Barrier after clear
    clear_barrier = MemoryBarrier(ACCESS_TRANSFER_WRITE_BIT, ACCESS_SHADER_WRITE_BIT)
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
    cmd_dispatch(cmd, gx, gy, gz)"""

content = content.replace(old_seed_dispatch, new_seed_dispatch)

old_seed_state = """    state.current_buffer_is_ping = true
    state.accumulated_steps = 0
    @debug "[HeatDiffusion] Seeded at ($seed_x, $seed_y, $seed_z)\" """

new_seed_state = """    state.current_buffer_is_ping = true
    state.accumulated_steps = 0
    
    state.seed_x = seed_x
    state.seed_y = seed_y
    state.seed_z = seed_z
    state.seed_radius = seed_radius
    
    R = seed_radius + 4
    xmin = max(1, seed_x - R)
    xmax = min(w, seed_x + R)
    ymin = max(1, seed_y - R)
    ymax = min(h, seed_y + R)
    zmin = max(1, seed_z - R)
    zmax = min(d, seed_z + R)
    state.current_box_min = (xmin, ymin, zmin)
    state.current_box_max = (xmax, ymax, zmax)
    
    @debug "[HeatDiffusion] Seeded at ($seed_x, $seed_y, $seed_z) radius=$seed_radius\" """

content = content.replace(old_seed_state, new_seed_state)

# Replace run_incremental_steps!
old_run = """function run_incremental_steps!(state::HeatDiffusionState, ctx, n_steps::Int;
                                 dt::Float32=0.16f0, theta::Float32=0.001f0, tau::Float32=0.0001f0)
    @assert state.is_initialized "Heat diffusion not initialized"
    @assert state.diffusivity_uploaded "Diffusivity not uploaded"
    n_steps <= 0 && return
    
    dev = ctx.device
    w, h, d = state.width, state.height, state.depth
    gx, gy, gz = cld(w, 8), cld(h, 8), cld(d, 8)
    
    cmd = state.compute_cmd_buffer
    unwrap(reset_command_buffer(cmd))
    begin_info = CommandBufferBeginInfo(flags = COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT)
    unwrap(begin_command_buffer(cmd, begin_info))
    
    barrier_compute = MemoryBarrier(C_NULL, ACCESS_SHADER_WRITE_BIT, ACCESS_SHADER_READ_BIT)
    
    # ── Diffusion steps (continue from current ping/pong state) ──
    cmd_bind_pipeline(cmd, PIPELINE_BIND_POINT_COMPUTE, state.pipeline_diffuse_step)
    dt_arr = Float32[dt]
    
    is_ping = state.current_buffer_is_ping
    for k in 1:n_steps
        if is_ping
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
        cmd_pipeline_barrier(cmd, [barrier_compute], BufferMemoryBarrier[], ImageMemoryBarrier[];
                             src_stage_mask=PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             dst_stage_mask=PIPELINE_STAGE_COMPUTE_SHADER_BIT)
        is_ping = !is_ping
    end
    
    # ── Extract mask from whichever buffer has the latest data ──
    cmd_bind_pipeline(cmd, PIPELINE_BIND_POINT_COMPUTE, state.pipeline_extract_mask)
    ds_extract = is_ping ? state.ds_extract_from_ping : state.ds_extract_from_pong
    cmd_bind_descriptor_sets(cmd, PIPELINE_BIND_POINT_COMPUTE, state.layout_extract_mask, 0, [ds_extract], [])
    extract_data = ExtractMaskPC[ExtractMaskPC(Float32(theta), Float32(tau), Int32(1), Int32(1))]
    GC.@preserve extract_data begin
        cmd_push_constants(cmd, state.layout_extract_mask, SHADER_STAGE_COMPUTE_BIT, 0, sizeof(extract_data), Ptr{Cvoid}(pointer(extract_data)))
    end
    cmd_dispatch(cmd, gx, gy, gz)
    
    cmd_pipeline_barrier(cmd, [barrier_compute], BufferMemoryBarrier[], ImageMemoryBarrier[];
                         src_stage_mask=PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         dst_stage_mask=PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT)
    
    unwrap(end_command_buffer(cmd))
    submit_info = SubmitInfo([], [], [cmd], [])
    unwrap(reset_fences(dev, [state.compute_fence]))
    unwrap(queue_submit(ctx.graphics_queue, [submit_info]; fence=state.compute_fence))
    unwrap(wait_for_fences(dev, [state.compute_fence], true, typemax(UInt64)))
    
    state.current_buffer_is_ping = is_ping
    state.accumulated_steps += n_steps
end"""

new_run = """function run_incremental_steps!(state::HeatDiffusionState, ctx, n_steps::Int;
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
        barrier_comp_to_xfer = MemoryBarrier(ACCESS_SHADER_WRITE_BIT, ACCESS_TRANSFER_READ_BIT)
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
        
        barrier_xfer_to_host = MemoryBarrier(ACCESS_TRANSFER_WRITE_BIT, ACCESS_HOST_READ_BIT)
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
end"""

content = content.replace(old_run, new_run)

# Add apply_box_mask_to_segvol!
new_func = """
# ─── Read bounding box mask directly to seg_vol ──────────────────────────

\"\"\"
    apply_box_mask_to_segvol!(state, seg_vol::AbstractArray{T, 3}, label_val::T; theta=0.001f0) where {T}

Applies the bounding-box mask retrieved from GPU directly into `seg_vol`.
Reads from the persistently mapped `staging_box_ptr` with ZERO allocations.
\"\"\"
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
"""

# Append to the end of the file, before `end # module VulkanHeatDiffusion`
content = content.replace("end # module VulkanHeatDiffusion", new_func + "\nend # module VulkanHeatDiffusion")

with open('src/display/Vulkan/VulkanHeatDiffusion.jl', 'w') as f:
    f.write(content)

print("Applied phase 2 patches successfully")
