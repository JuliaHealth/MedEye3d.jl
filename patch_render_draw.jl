path = "src/display/Vulkan/VulkanRender.jl"
content = read(path, String)
content = replace(content, "n_vertices = length(panel.vector_vertices) ÷ 6" => "n_vertices = length(force_verts) ÷ 6")
write(path, content)
