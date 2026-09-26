path = "src/display/Vulkan/VulkanRender.jl"
content = read(path, String)
content = replace(content, """
            if true
                open("/tmp/medeye_measure_pc.log", "a") do f
                    println(f, "vector_vertices len: ", length(panel.vector_vertices))
                    println(f, "push constants: ", pc_data)
                end
            end
""" => """
            open("/tmp/medeye_measure_pc.log", "a") do f
                println(f, "vector_vertices len: ", length(panel.vector_vertices))
                println(f, "push constants: ", pc_data)
            end
""")
write(path, content)
