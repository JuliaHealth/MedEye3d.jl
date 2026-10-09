content = read("src/display/GLFW/MakieEventHandlers.jl", String)

# In reactToHeatGDTStart
content = replace(content, """
                vk_tex = obj.vulkanTextures[mask_idx]
                slice_dim = state.onScrollData.dimensionToScroll
                slice_idx = state.currentlyDispDat.sliceNumber - 1
                VulkanHeatDiffusion.render_slice_to_texture!(heat_state, vk_ctx, vk_tex, slice_dim, slice_idx, active_id; theta=theta_val)
                if obj.vulkanPipelineState !== nothing
                    obj.vulkanPipelineState.ubo_dirty = true
                end
""" => """
                vk_tex = obj.vulkanTextures[mask_idx]
                try
                    slice_dim = state.onScrollData.dimensionToScroll
                    slice_idx = state.currentlyDispDat.sliceNumber - 1
                    VulkanHeatDiffusion.render_slice_to_texture!(heat_state, vk_ctx, vk_tex, slice_dim, slice_idx, active_id; theta=theta_val)
                    if obj.vulkanPipelineState !== nothing
                        obj.vulkanPipelineState.ubo_dirty = true
                    end
                catch e
                    # Ignore slice extraction errors on secondary panels so they don't break the main loop
                    @debug "[Heat-GDT GPU] Ignoring panel error: \$e"
                end
""")

# In reactToHeatGDTTick
content = replace(content, """
                vk_tex = obj.vulkanTextures[mask_idx]
                
                # 1=sagittal(X), 2=coronal(Y), 3=axial(Z)
                slice_dim = state.onScrollData.dimensionToScroll
                slice_idx = state.currentlyDispDat.sliceNumber - 1  # 0-indexed for shader
                
                active_id = UInt32(current_active_lesion_id[] > 0 ? current_active_lesion_id[] : 1)
                
                VulkanHeatDiffusion.render_slice_to_texture!(heat_state, vk_ctx, vk_tex, slice_dim, slice_idx, active_id; theta=theta_val)
                
                if obj.vulkanPipelineState !== nothing
                    obj.vulkanPipelineState.ubo_dirty = true
                end
""" => """
                vk_tex = obj.vulkanTextures[mask_idx]
                
                try
                    # 1=sagittal(X), 2=coronal(Y), 3=axial(Z)
                    slice_dim = state.onScrollData.dimensionToScroll
                    slice_idx = state.currentlyDispDat.sliceNumber - 1  # 0-indexed for shader
                    
                    active_id = UInt32(current_active_lesion_id[] > 0 ? current_active_lesion_id[] : 1)
                    
                    VulkanHeatDiffusion.render_slice_to_texture!(heat_state, vk_ctx, vk_tex, slice_dim, slice_idx, active_id; theta=theta_val)
                    
                    if obj.vulkanPipelineState !== nothing
                        obj.vulkanPipelineState.ubo_dirty = true
                    end
                catch e
                    @debug "[Heat-GDT GPU] Ignoring panel error: \$e"
                end
""")

write("src/display/GLFW/MakieEventHandlers.jl", content)
