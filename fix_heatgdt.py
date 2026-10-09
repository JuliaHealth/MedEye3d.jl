import re

with open("src/display/GLFW/MakieEventHandlers.jl", "r") as f:
    code = f.read()

# Fix UBO minAndMaxValue inside reactToHeatGDTStart
# Right after: _vk_heat_active[] = true
# Before: _vk_heat_last_dispatch[] = time()

ubo_update = """
    # Update UBO minAndMaxValue to include the active ID so the shader does not filter it out
    active_id = UInt32(current_active_lesion_id[] > 0 ? current_active_lesion_id[] : 1)
    for state in stateObjects
        obj = state.mainForDisplayObjects
        if obj !== nothing && obj.vulkanTextures !== nothing
            for vk_tex in obj.vulkanTextures
                if vk_tex.info.name == "Mask" || vk_tex.info.name == "segmentation"
                    if vk_tex.info.minAndMaxValue[1] > active_id || vk_tex.info.minAndMaxValue[2] < active_id
                        vk_tex.info.minAndMaxValue[1] = min(vk_tex.info.minAndMaxValue[1], Float32(active_id))
                        vk_tex.info.minAndMaxValue[2] = max(vk_tex.info.minAndMaxValue[2], Float32(active_id))
                        if obj.vulkanPipelineState !== nothing
                            obj.vulkanPipelineState.ubo_dirty = true
                        end
                    end
                end
            end
        end
    end
"""

# Find _vk_heat_active[] = true
code = re.sub(
    r"(_vk_heat_active\[\] = true\n)",
    r"\1" + ubo_update,
    code
)

# Fix slice_dim in reactToHeatGDTStart
# Replace slice_dim = state.onScrollData.dimensionToScroll with panel_idx mapped value

slice_dim_patch = """
                    panel_idx = findfirst(s -> s === state, stateObjects)
                    slice_dim = if panel_idx in (3, 8)
                        1 # Sagittal
                    elseif panel_idx in (4, 9)
                        2 # Coronal
                    else
                        3 # Axial
                    end
"""

code = re.sub(
    r"slice_dim = state\.onScrollData\.dimensionToScroll\n\s*slice_idx = state\.currentlyDispDat\.sliceNumber - 1",
    slice_dim_patch + "                    slice_idx = state.currentlyDispDat.sliceNumber - 1",
    code
)

# Fix M key in test_inject.txt
m_key_patch = """
                        elseif cmd == "M"
                            put!(_ch, MakieEvents.ToggleSingleMultiLesionEvent())
                            println("[TEST-INJECT] Sent ToggleSingleMultiLesionEvent"); flush(stdout)
"""

# Insert M key before STOP
code = re.sub(
    r"(elseif cmd == \"STOP\")",
    m_key_patch.strip("\n") + "\n                        \\1",
    code
)

# Modify theta ramp for HEAT_DISPLAY_RAMP_START to be lower?
# "Adjust HEAT_DISPLAY_RAMP_START and HEAT_DISPLAY_RAMP_RATE in MakieEventHandlers.jl"
# Let's see what is their current value.
code = re.sub(r"const HEAT_DISPLAY_RAMP_START = 0\.5f0", "const HEAT_DISPLAY_RAMP_START = 0.1f0", code)
code = re.sub(r"const HEAT_DISPLAY_RAMP_RATE = 2\.0f0", "const HEAT_DISPLAY_RAMP_RATE = 4.0f0", code)

with open("src/display/GLFW/MakieEventHandlers.jl", "w") as f:
    f.write(code)

