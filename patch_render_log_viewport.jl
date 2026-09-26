path = "src/display/Vulkan/VulkanRender.jl"
content = read(path, String)
content = replace(content, """
            open("/tmp/medeye_measure_pc.log", "a") do f
                println(f, "vector_vertices len: ", length(panel.vector_vertices))
                println(f, "push constants: ", pc_data)
            end
""" => """
            open("/tmp/medeye_measure_pc.log", "a") do f
                println(f, "vector_vertices len: ", length(panel.vector_vertices))
                println(f, "push constants: ", pc_data)
                println(f, "viewport: ", panel.viewport_x, ", ", panel.viewport_y, ", ", panel.viewport_w, ", ", panel.viewport_h)
            end
""")
write(path, content)
