import re

with open("src/display/GLFW/MakieEventHandlers.jl", "r") as f:
    code = f.read()

target = """        if mask !== nothing && seg_vol !== nothing
            InferenceClient.insert_patch!(seg_vol, mask, cx, cy, cz, label_val=Float32(active_id))
            @info "HelpNet/NNInteractive segmented $(count(mask .> 0)) patch voxels for lesion $active_id at ($cx, $cy, $cz)."
"""

replacement = """        if mask !== nothing && seg_vol !== nothing
            InferenceClient.insert_patch!(seg_vol, mask, cx, cy, cz, label_val=Float32(active_id))
            @info "HelpNet/NNInteractive segmented $(count(mask .> 0)) patch voxels for lesion $active_id at ($cx, $cy, $cz)."
            
            if haskey(bone_subsegments_cache, active_id)
                delete!(bone_subsegments_cache, active_id)
                @info "Invalidated bone subsegments cache for lesion $active_id."
            end
"""

code = code.replace(target, replacement)

target2 = """            # Propagate canonical mask volume to all panels
            for (p_idx, st) in enumerate(stateObjects)
                for scrDat in st.onScrollData.dataToScroll
                    if scrDat.name == "manualModif" || scrDat.name == "segmentation" || scrDat.name == "Mask"
"""
replacement2 = """            # Propagate canonical mask volume to all panels
            for (p_idx, st) in enumerate(stateObjects)
                for scrDat in st.onScrollData.dataToScroll
                    if scrDat.name == "manualModif"
                        fill!(scrDat.dat, 0.0f0)
                        @info "Filled scrDat $(scrDat.name) with zeros."
                    elseif scrDat.name == "segmentation" || scrDat.name == "Mask"
"""

code = code.replace(target2, replacement2)

with open("src/display/GLFW/MakieEventHandlers.jl", "w") as f:
    f.write(code)
