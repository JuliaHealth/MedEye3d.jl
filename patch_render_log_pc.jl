path = "src/display/Vulkan/VulkanRender.jl"
content = read(path, String)
content = replace(content, """
            ptr = unwrap(Vulkan.map_memory(ctx.device, mem, 0, data_size))
""" => """
            if rand() < 0.05
                open("/tmp/medeye_measure_pc.log", "a") do f
                    println(f, "vector_vertices len: ", length(panel.vector_vertices))
                    println(f, "push constants: ", pc_data)
                end
            end
            ptr = unwrap(Vulkan.map_memory(ctx.device, mem, 0, data_size))
""")
write(path, content)
