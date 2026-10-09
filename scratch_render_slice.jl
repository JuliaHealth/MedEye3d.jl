function render_slice_to_texture!(state::HeatDiffusionState, ctx, tex::Any, slice_dim::Int, slice_idx::Int, active_id::UInt32; theta::Float32=0.001f0)
    dev = ctx.device
    is_ping = state.current_buffer_is_ping
    ds = is_ping ? state.ds_slice_ping : state.ds_slice_pong
    
    # Update descriptor set binding 1 with tex.view
    # Wait, we need to import or qualify Vulkan types.
    di = DescriptorImageInfo(state.dummy_sampler, tex.view, IMAGE_LAYOUT_GENERAL)
    write = WriteDescriptorSet(ds, 1, 0, DESCRIPTOR_TYPE_STORAGE_IMAGE, [di], [], [])
    update_descriptor_sets(dev, [write], [])
    
    cmd = state.compute_cmd_buffer
    unwrap(reset_command_buffer(cmd))
    begin_info = CommandBufferBeginInfo(flags = COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT)
    unwrap(begin_command_buffer(cmd, begin_info))
    
    # Transition texture to GENERAL
    barrier_to_general = ImageMemoryBarrier(
        C_NULL, ACCESS_SHADER_WRITE_BIT,
        IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, IMAGE_LAYOUT_GENERAL,
        0, 0, tex.image, ImageSubresourceRange(IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1)
    )
    cmd_pipeline_barrier(cmd, MemoryBarrier[], BufferMemoryBarrier[], [barrier_to_general];
                         src_stage_mask=PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         dst_stage_mask=PIPELINE_STAGE_COMPUTE_SHADER_BIT)
    
    # Bind pipeline
    cmd_bind_pipeline(cmd, PIPELINE_BIND_POINT_COMPUTE, state.pipeline_slice)
    cmd_bind_descriptor_sets(cmd, PIPELINE_BIND_POINT_COMPUTE, state.layout_slice, 0, [ds], [])
    
    # Push constants
    # pc: float theta, int slice_dim, int slice_idx, uint active_id
    pc_data = [theta, Float32(slice_dim), Float32(slice_idx), Float32(reinterpret(Float32, active_id))]
    # Wait! Push constants must exactly match. 
    # int slice_dim, int slice_idx, uint active_id. It's safer to use a struct.
    # We can define SlicePC struct, or just use Int32 array with reinterpret.
    # struct SlicePC theta::Float32; slice_dim::Int32; slice_idx::Int32; active_id::UInt32 end
    # Or just use an array of UInt32
    pc_u32 = UInt32[reinterpret(UInt32, theta), UInt32(slice_dim), UInt32(slice_idx), active_id]
    
    GC.@preserve pc_u32 begin
        cmd_push_constants(cmd, state.layout_slice, SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc_u32), Ptr{Cvoid}(pointer(pc_u32)))
    end
    
    # Dispatch
    gx = cld(tex.width, 8)
    gy = cld(tex.height, 8)
    cmd_dispatch(cmd, gx, gy, 1)
    
    # Transition back to SHADER_READ_ONLY_OPTIMAL
    barrier_to_read = ImageMemoryBarrier(
        ACCESS_SHADER_WRITE_BIT, ACCESS_SHADER_READ_BIT,
        IMAGE_LAYOUT_GENERAL, IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        0, 0, tex.image, ImageSubresourceRange(IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1)
    )
    cmd_pipeline_barrier(cmd, MemoryBarrier[], BufferMemoryBarrier[], [barrier_to_read];
                         src_stage_mask=PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         dst_stage_mask=PIPELINE_STAGE_FRAGMENT_SHADER_BIT)
                         
    unwrap(end_command_buffer(cmd))
    submit_info = SubmitInfo([], [], [cmd], [])
    unwrap(reset_fences(dev, [state.compute_fence]))
    unwrap(queue_submit(ctx.graphics_queue, [submit_info]; fence=state.compute_fence))
    unwrap(wait_for_fences(dev, [state.compute_fence], true, typemax(UInt64)))
end
