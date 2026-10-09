import re

with open("src/display/Vulkan/VulkanHeatDiffusion.jl", "r") as f:
    code = f.read()

# Update pc_diffuse construction in run_incremental_steps!
old_pc = """    pc_diffuse = DiffuseStepPC(
        dt, Int32(1), Int32(0), Int32(0),
        box_min_glsl[1], box_min_glsl[2], box_min_glsl[3], Int32(0),
        box_max_glsl[1], box_max_glsl[2], box_max_glsl[3], Int32(0)
    )"""

new_loop = """    is_ping = state.current_buffer_is_ping
    for k in 1:n_steps
        current_step = UInt32(state.accumulated_steps + k)
        pc_diffuse = DiffuseStepPC(
            dt, Int32(1), Int32(0), Int32(0),
            box_min_glsl[1], box_min_glsl[2], box_min_glsl[3], Int32(0),
            box_max_glsl[1], box_max_glsl[2], box_max_glsl[3], Int32(0),
            theta, current_step
        )
        if is_ping"""

code = code.replace(old_pc, "")
code = code.replace("    is_ping = state.current_buffer_is_ping\n    for k in 1:n_steps\n        if is_ping", new_loop)

with open("src/display/Vulkan/VulkanHeatDiffusion.jl", "w") as f:
    f.write(code)

