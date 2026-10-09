import re

with open("src/display/GLFW/MakieEventHandlers.jl", "r") as f:
    code = f.read()

ubo_update_bad = """    # Update UBO minAndMaxValue to include the active ID so the shader does not filter it out
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
    end"""

ubo_update_good = """    # Update UBO minAndMaxValue to include the active ID so the shader does not filter it out
    active_id = UInt32(current_active_lesion_id[] > 0 ? current_active_lesion_id[] : 1)
    for state in stateObjects
        obj = state.mainForDisplayObjects
        if obj !== nothing && obj.listOfTextSpecifications !== nothing
            for ts in obj.listOfTextSpecifications
                if ts.name == "Mask" || ts.name == "segmentation"
                    if !isempty(ts.minAndMaxValue)
                        if ts.minAndMaxValue[1] > active_id || ts.minAndMaxValue[2] < active_id
                            ts.minAndMaxValue[1] = min(ts.minAndMaxValue[1], Float32(active_id))
                            ts.minAndMaxValue[2] = max(ts.minAndMaxValue[2], Float32(active_id))
                            if obj.vulkanPipelineState !== nothing
                                obj.vulkanPipelineState.ubo_dirty = true
                            end
                        end
                    end
                end
            end
        end
    end"""

code = code.replace(ubo_update_bad, ubo_update_good)

with open("src/display/GLFW/MakieEventHandlers.jl", "w") as f:
    f.write(code)

