function reactToHeatGDTTick(::MakieEvents.HeatGDTTickEvent, stateObjects::Vector{StateDataFields})\
    if !_vk_heat_active[]; return; end\
    \
    # ── Debounce: skip if too soon since last dispatch ──\
    now = time()\
    if (now - _vk_heat_last_dispatch[]) < HEAT_TICK_MIN_INTERVAL_S\
        return\
    end\
    _vk_heat_last_dispatch[] = now\
    \
    heat_state = _vk_heat_state[]\
    vk_ctx = _vk_heat_ctx[]\
    if heat_state === nothing || vk_ctx === nothing; return; end\
    \
    try\
        # ── Run a larger batch of diffusion steps on GPU (100% GPU) ──\
        steps_per_tick = 50\
        theta_val = heatgdt_theta[]\
        \
        t0 = time_ns()\
        VulkanHeatDiffusion.run_incremental_steps!(heat_state, vk_ctx, steps_per_tick;\
            theta=theta_val, copy_box_to_cpu=false)\
        t_gpu = time_ns()\
        _vk_heat_total_steps[] += steps_per_tick\
        \
        # ── 100% GPU Overlay Rendering ──\
        # Directly slice the 3D heat field onto the 2D UI texture for each panel.\
        for state in stateObjects\
            obj = state.mainForDisplayObjects\
            if obj === nothing || obj.vulkanTextures === nothing || isempty(obj.vulkanTextures)\
                continue\
            end\
            \
            # Find Mask texture index\
            mask_idx = findfirst(d -> d.name == "Mask" || d.name == "segmentation", state.onScrollData.dataToScroll)\
            if mask_idx !== nothing && mask_idx <= length(obj.vulkanTextures)\
                vk_tex = obj.vulkanTextures[mask_idx]\
                \
                # 1=sagittal(X), 2=coronal(Y), 3=axial(Z)\
                slice_dim = state.currentlyDispDat.dimensionToScroll\
                slice_idx = state.currentlyDispDat.sliceNumber - 1  # 0-indexed for shader\
                \
                active_id = UInt32(current_active_lesion_id[] > 0 ? current_active_lesion_id[] : 1)\
                \
                VulkanHeatDiffusion.render_slice_to_texture!(heat_state, vk_ctx, vk_tex, slice_dim, slice_idx, active_id; theta=theta_val)\
                \
                if obj.vulkanPipelineState !== nothing\
                    obj.vulkanPipelineState.ubo_dirty = true\
                end\
            end\
        end\
        t_render = time_ns()\
        \
        gpu_ms = (t_gpu - t0) / 1e6\
        render_ms = (t_render - t_gpu) / 1e6\
        total_ms = (t_render - t0) / 1e6\
        \
        log_line = "[Heat-GDT GPU Tick] steps=$(steps_per_tick), total=$(round(total_ms, digits=2))ms (gpu=$(round(gpu_ms, digits=2)), slice_render=$(round(render_ms, digits=2)))"\
        if PERF_LOG[]\
            println(log_line)\
        end\
        open("heatgdt_perf.log", "a") do f\
            println(f, log_line)\
        end\
    catch e\
        @warn "[Heat-GDT GPU] Tick failed" exception=(e, catch_backtrace())\
    end\
end
