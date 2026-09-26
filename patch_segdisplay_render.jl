path = "src/display/GLFW/SegmentationDisplay.jl"
content = read(path, String)
content = replace(content, """
                        w = (panel_idx > 5 && m2_vk[] !== nothing) ? Float32(m2_vk[].swapchain_extent.width) : Float32(obj.vulkanCtx.width)
                        h = (panel_idx > 5 && m2_vk[] !== nothing) ? Float32(m2_vk[].swapchain_extent.height) : Float32(obj.vulkanCtx.height)
                        
                        panel = VulkanRender.PanelRenderData(
                            obj.vulkanPipelineState,
                            push_consts,
                            Float32(0), Float32(0), w, h
                        )
""" => """
                        w = (panel_idx > 5 && m2_vk[] !== nothing) ? Float32(m2_vk[].swapchain_extent.width) : Float32(obj.vulkanCtx.width)
                        h = (panel_idx > 5 && m2_vk[] !== nothing) ? Float32(m2_vk[].swapchain_extent.height) : Float32(obj.vulkanCtx.height)
                        
                        vector_vertices = Measurements.compute_measurement_vertices(obj.measurements, stateInstances, panel_idx)
                        line_verts = Measurements.compute_line_vertices(obj.line_measurements, stateInstances, panel_idx)
                        append!(vector_vertices, line_verts)
                        
                        # Ensure VBO is large enough for this frame's vector data
                        vbo_for_panel = obj.vulkanVectorVBO
                        data_size = sizeof(Float32) * length(vector_vertices)
                        if data_size > 0 && obj.vulkanCtx !== nothing
                            vbo_for_panel = VulkanRender.ensure_vector_vbo!(obj.vulkanCtx, vbo_for_panel, data_size)
                            obj.vulkanVectorVBO = vbo_for_panel
                        end
                        
                        panel = VulkanRender.PanelRenderData(
                            obj.vulkanPipelineState,
                            obj.vulkanVectorPipelineState,
                            push_consts,
                            Float32(0), Float32(0), w, h,
                            vector_vertices,
                            vbo_for_panel
                        )
""")
write(path, content)
