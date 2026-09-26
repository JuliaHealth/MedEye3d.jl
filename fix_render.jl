path = "src/display/Vulkan/VulkanRender.jl"
content = read(path, String)
content = replace(content, """
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
""" => "")
write(path, content)
