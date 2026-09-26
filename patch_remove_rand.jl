path = "src/display/GLFW/SegmentationDisplay.jl"
content = read(path, String)
content = replace(content, "if length(obj.measurements) > 0 && rand() < 0.05" => "if length(obj.measurements) > 0")
write(path, content)

path = "src/display/Vulkan/VulkanRender.jl"
content = read(path, String)
content = replace(content, "if rand() < 0.05" => "if true")
write(path, content)
