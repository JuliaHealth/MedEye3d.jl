path = "src/display/GLFW/SegmentationDisplay.jl"
content = read(path, String)
content = replace(content, """
                        append!(vector_vertices, line_verts)
                        
                        if length(obj.measurements) > 0
""" => """
                        append!(vector_vertices, line_verts)
                        
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
                        
                        if length(obj.measurements) > 0
""")
write(path, content)
