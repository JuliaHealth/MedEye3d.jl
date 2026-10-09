import re

with open('src/display/GLFW/MakieEventHandlers.jl', 'r') as f:
    content = f.read()

# 1. Update reactToHeatGDTStart
old_start = """    # Seed the GPU heat field at cursor position (no diffusion steps yet)
    try
        println("[Heat-GDT START] Calling seed_only! at ($cx,$cy,$cz)..."); flush(stdout)
        VulkanHeatDiffusion.seed_only!(heat_state, vk_ctx, cx, cy, cz)
        set_ai_status!("[Heat-GDT GPU] Growing from ($cx,$cy,$cz)...")
        println("[Heat-GDT START] seed_only! SUCCESS — GPU diffusion active"); flush(stdout)
    catch e
        _vk_heat_active[] = false
        set_ai_status!("[Heat-GDT GPU] Seed failed: $(sprint(showerror, e))")
        @warn "[Heat-GDT GPU] seed_only! failed" exception=(e, catch_backtrace())
        println("[Heat-GDT START] BLOCKED: seed_only! threw: $(sprint(showerror, e))"); flush(stdout)
        return
    end
    
    # Start an async loop"""

new_start = """    # Seed the GPU heat field at cursor position (no diffusion steps yet)
    try
        println("[Heat-GDT START] Calling seed_only! at ($cx,$cy,$cz)..."); flush(stdout)
        VulkanHeatDiffusion.seed_only!(heat_state, vk_ctx, cx, cy, cz; seed_radius=3)
        set_ai_status!("[Heat-GDT GPU] Growing from ($cx,$cy,$cz)...")
        println("[Heat-GDT START] seed_only! SUCCESS — GPU diffusion active"); flush(stdout)
        
        # Run 6 initial steps immediately so the lesion appears instantly on click
        theta_val = heatgdt_theta[]
        VulkanHeatDiffusion.run_incremental_steps!(heat_state, vk_ctx, 6; theta=theta_val, copy_box_to_cpu=true)
        _vk_heat_total_steps[] += 6
        
        active_id = current_active_lesion_id[] > 0 ? current_active_lesion_id[] : 1
        if seg_vol !== nothing
            label_val = eltype(seg_vol)(active_id)
            VulkanHeatDiffusion.apply_box_mask_to_segvol!(heat_state, seg_vol, label_val; theta=theta_val)
        end
        for state in stateObjects
            state.isSliceChanged = true
        end
    catch e
        _vk_heat_active[] = false
        set_ai_status!("[Heat-GDT GPU] Seed failed: $(sprint(showerror, e))")
        @warn "[Heat-GDT GPU] seed_only! failed" exception=(e, catch_backtrace())
        println("[Heat-GDT START] BLOCKED: seed_only! threw: $(sprint(showerror, e))"); flush(stdout)
        return
    end
    
    # Start an async loop"""

content = content.replace(old_start, new_start)

# 2. Update reactToHeatGDTTick
old_tick = """        # ── Run a small batch of diffusion steps on GPU ──
        steps_per_tick = 4
        theta_val = heatgdt_theta[]
        
        t0 = time_ns()
        VulkanHeatDiffusion.run_incremental_steps!(heat_state, vk_ctx, steps_per_tick;
            theta=theta_val)
        t_gpu = time_ns()
        _vk_heat_total_steps[] += steps_per_tick
        
        # ── Read full 3D mask from GPU ──
        mask_3d = VulkanHeatDiffusion.read_mask_to_cpu(heat_state, vk_ctx)
        t_read = time_ns()
        
        # ── Write to seg_vol (the shared 3D array all panels slice from) ──
        _apply_gpu_mask_to_segvol!(mask_3d, stateObjects)
        t_apply = time_ns()
        
        # ── Mark ALL panels dirty → axial + sagittal + coronal all refresh ──
        for state in stateObjects
            state.isSliceChanged = true
        end
        t_end = time_ns()
        
        # Benchmark logging
        gpu_ms = (t_gpu - t0) / 1e6
        read_ms = (t_read - t_gpu) / 1e6
        apply_ms = (t_apply - t_read) / 1e6
        render_ms = (t_end - t_apply) / 1e6
        total_ms = (t_end - t0) / 1e6
        
        log_line = "[Heat-GDT GPU Tick] steps=$(steps_per_tick), total=$(round(total_ms, digits=2))ms (gpu=$(round(gpu_ms, digits=2)), read=$(round(read_ms, digits=2)), apply=$(round(apply_ms, digits=2)), trigger_render=$(round(render_ms, digits=2)))" """

new_tick = """        # ── Run a small batch of diffusion steps on GPU ──
        steps_per_tick = 8
        theta_val = heatgdt_theta[]
        
        t0 = time_ns()
        VulkanHeatDiffusion.run_incremental_steps!(heat_state, vk_ctx, steps_per_tick;
            theta=theta_val, copy_box_to_cpu=true)
        t_gpu = time_ns()
        _vk_heat_total_steps[] += steps_per_tick
        
        # ── Write bounding box to seg_vol ──
        tp1 = stateObjects[1]
        seg_vol = nothing
        for dat in tp1.onScrollData.dataToScroll
            if dat.name == "Mask" || dat.name == "segmentation"
                seg_vol = dat.dat
                break
            end
        end
        if seg_vol !== nothing
            active_id = current_active_lesion_id[] > 0 ? current_active_lesion_id[] : 1
            label_val = eltype(seg_vol)(active_id)
            VulkanHeatDiffusion.apply_box_mask_to_segvol!(heat_state, seg_vol, label_val; theta=theta_val)
        end
        t_apply = time_ns()
        
        # ── Mark ALL panels dirty → axial + sagittal + coronal all refresh ──
        for state in stateObjects
            state.isSliceChanged = true
        end
        t_end = time_ns()
        
        # Benchmark logging
        gpu_ms = (t_gpu - t0) / 1e6
        apply_ms = (t_apply - t_gpu) / 1e6
        render_ms = (t_end - t_apply) / 1e6
        total_ms = (t_end - t0) / 1e6
        
        log_line = "[Heat-GDT GPU Tick] steps=$(steps_per_tick), total=$(round(total_ms, digits=2))ms (gpu=$(round(gpu_ms, digits=2)), apply=$(round(apply_ms, digits=2)), trigger_render=$(round(render_ms, digits=2)))" """

content = content.replace(old_tick, new_tick)

with open('src/display/GLFW/MakieEventHandlers.jl', 'w') as f:
    f.write(content)

print("Applied Makie patches successfully")
