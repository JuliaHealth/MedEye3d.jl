path = "src/display/Vulkan/VulkanRender.jl"
content = read(path, String)
content = replace(content, "length(force_verts)" => "length(panel.vector_vertices)")
content = replace(content, "pointer(force_verts)" => "pointer(panel.vector_vertices)")
write(path, content)
