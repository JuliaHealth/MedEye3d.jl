import re

with open("src/display/GLFW/MakieEventHandlers.jl", "r") as f:
    code = f.read()

old_block = """        # Artificial slow-down at the beginning to prevent instant explosion
        # Ramp up speed over the first ~500ms based on total accumulated steps
        curr_steps = heat_state.accumulated_steps
        ramp_factor = 1.0 + (curr_steps / 80.0)^1.8 * 20.0
        steps_per_tick = clamp(round(Int, ramp_factor), 1, perf_max_steps)
        
        theta_val = heatgdt_theta[]
        
        t0 = time_ns()
        VulkanHeatDiffusion.run_incremental_steps!(heat_state, vk_ctx, steps_per_tick;
            theta=theta_val, copy_box_to_cpu=false)
        t_gpu = time_ns()
        _vk_heat_total_steps[] += steps_per_tick"""

new_block = """        # Compute as fast as possible on the GPU (run full perf_max_steps)
        steps_per_tick = perf_max_steps
        
        # Artificial slow-down for the UI rendering to prevent instant explosion
        # Ramp up speed based on total displayed steps so it caches ahead and catches up
        curr_display = _vk_heat_display_steps[]
        ramp_factor = 1.0 + (curr_display / 80.0)^1.8 * 20.0
        display_inc = clamp(round(Int, ramp_factor), 1, perf_max_steps)
        
        theta_val = heatgdt_theta[]
        
        t0 = time_ns()
        # Note: pass display_step if run_incremental_steps! needed it, but we handle extraction later or via slice shader
        VulkanHeatDiffusion.run_incremental_steps!(heat_state, vk_ctx, steps_per_tick;
            theta=theta_val, copy_box_to_cpu=false)
        t_gpu = time_ns()
        
        _vk_heat_total_steps[] += steps_per_tick
        _vk_heat_display_steps[] = min(_vk_heat_total_steps[], _vk_heat_display_steps[] + display_inc)"""

if old_block in code:
    with open("src/display/GLFW/MakieEventHandlers.jl", "w") as f:
        f.write(code.replace(old_block, new_block))
    print("Replaced block 1")
else:
    print("Block 1 not found!")

old_render = """                    VulkanHeatDiffusion.render_slice_to_texture!(heat_state, vk_ctx, vk_tex, slice_dim, slice_idx, active_id; theta=theta_val)"""

new_render = """                    display_step = UInt32(_vk_heat_display_steps[])
                    VulkanHeatDiffusion.render_slice_to_texture!(heat_state, vk_ctx, vk_tex, slice_dim, slice_idx, active_id, display_step; theta=theta_val)"""

code = code.replace(old_block, new_block)

if old_render in code:
    with open("src/display/GLFW/MakieEventHandlers.jl", "w") as f:
        f.write(code.replace(old_render, new_render))
    print("Replaced block 2")
else:
    print("Block 2 not found!")

