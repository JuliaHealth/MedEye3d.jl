import re

with open('src/display/GLFW/MakieEventHandlers.jl', 'r') as f:
    content = f.read()

# Replace the entire reactToHeatGDTTick function
pattern = r"function reactToHeatGDTTick\(::MakieEvents\.HeatGDTTickEvent, stateObjects::Vector\{StateDataFields\}\).*?catch e\s+@warn \"\[Heat-GDT GPU\] Tick failed\" exception=\(e, catch_backtrace\(\)\)\s+end\s+end"

new_tick = """function reactToHeatGDTTick(::MakieEvents.HeatGDTTickEvent, stateObjects::Vector{StateDataFields})
    if !_vk_heat_active[]; return; end
    
    # ── Debounce: skip if too soon since last dispatch ──
    now = time()
    if (now - _vk_heat_last_dispatch[]) < HEAT_TICK_MIN_INTERVAL_S
        return
    end
    _vk_heat_last_dispatch[] = now
    
    heat_state = _vk_heat_state[]
    vk_ctx = _vk_heat_ctx[]
    if heat_state === nothing || vk_ctx === nothing; return; end
    
    try
        # ── Run a small batch of diffusion steps on GPU ──
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
        
        log_line = "[Heat-GDT GPU Tick] steps=$(steps_per_tick), total=$(round(total_ms, digits=2))ms (gpu=$(round(gpu_ms, digits=2)), apply=$(round(apply_ms, digits=2)), trigger_render=$(round(render_ms, digits=2)))"
        if PERF_LOG[]
            println(log_line)
        end
        open("heatgdt_perf.log", "a") do f
            println(f, log_line)
        end
    catch e
        @warn "[Heat-GDT GPU] Tick failed" exception=(e, catch_backtrace())
    end
end"""

if re.search(pattern, content, re.DOTALL):
    content = re.sub(pattern, new_tick, content, flags=re.DOTALL)
    print("Patch applied")
else:
    print("Pattern not found")

with open('src/display/GLFW/MakieEventHandlers.jl', 'w') as f:
    f.write(content)
