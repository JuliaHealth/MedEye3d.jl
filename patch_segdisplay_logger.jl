path = "src/display/GLFW/SegmentationDisplay.jl"
content = read(path, String)
content = replace(content, """
                        vector_vertices = Measurements.compute_measurement_vertices(obj.measurements, stateInstances, panel_idx)
                        line_verts = Measurements.compute_line_vertices(obj.line_measurements, stateInstances, panel_idx)
                        append!(vector_vertices, line_verts)
""" => """
                        vector_vertices = Measurements.compute_measurement_vertices(obj.measurements, stateInstances, panel_idx)
                        line_verts = Measurements.compute_line_vertices(obj.line_measurements, stateInstances, panel_idx)
                        append!(vector_vertices, line_verts)
                        
                        if length(obj.measurements) > 0 && rand() < 0.05
                            open("/tmp/medeye_measure.log", "a") do f
                                println(f, "PANEL \$panel_idx has \$(length(obj.measurements)) measurements. Generates \$(length(vector_vertices)) floats.")
                            end
                        end
""")
write(path, content)
