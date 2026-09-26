path1 = "src/display/Vulkan/VulkanRender.jl"
content1 = read(path1, String)
content1 = replace(content1, """
            # FORCE DEBUG QUAD FOR TESTING
            force_verts = copy(panel.vector_vertices)
            append!(force_verts, Float32[
                0.25f0, 0.25f0, 1.0f0, 0.0f0, 0.0f0, 1.0f0,
                0.75f0, 0.25f0, 1.0f0, 0.0f0, 0.0f0, 1.0f0,
                0.25f0, 0.75f0, 1.0f0, 0.0f0, 0.0f0, 1.0f0,
                0.25f0, 0.75f0, 1.0f0, 0.0f0, 0.0f0, 1.0f0,
                0.75f0, 0.25f0, 1.0f0, 0.0f0, 0.0f0, 1.0f0,
                0.75f0, 0.75f0, 1.0f0, 0.0f0, 0.0f0, 1.0f0
            ])
            
            # Draw
            n_vertices = length(force_verts) ÷ 6
""" => """
            # Draw
            n_vertices = length(panel.vector_vertices) ÷ 6
""")
write(path1, content1)

path2 = "src/display/GLFW/SegmentationDisplay.jl"
content2 = read(path2, String)
content2 = replace(content2, """
                        if panel_idx == 1
                            append!(vector_vertices, Float32[
                                0.25f0, 0.25f0, 0.0f0, 1.0f0, 0.0f0, 1.0f0,
                                0.75f0, 0.25f0, 0.0f0, 1.0f0, 0.0f0, 1.0f0,
                                0.25f0, 0.75f0, 0.0f0, 1.0f0, 0.0f0, 1.0f0,
                                0.25f0, 0.75f0, 0.0f0, 1.0f0, 0.0f0, 1.0f0,
                                0.75f0, 0.25f0, 0.0f0, 1.0f0, 0.0f0, 1.0f0,
                                0.75f0, 0.75f0, 0.0f0, 1.0f0, 0.0f0, 1.0f0
                            ])
                        end
""" => "")
write(path2, content2)

path3 = "src/structs/Measurements.jl"
content3 = read(path3, String)
content3 = replace(content3, "t = 0.015f0" => "t = 0.040f0")
write(path3, content3)

