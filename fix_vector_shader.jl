path = "src/display/Vulkan/VulkanShaders.jl"
content = read(path, String)
content = replace(content, "vec2 transformedUV = (aImageUV - 0.5 - pc.uvOffset) / pc.uvScale + 0.5;" => "vec2 transformedUV = (aImageUV - 0.5) * pc.uvScale + 0.5 + pc.uvOffset;")
write(path, content)
println("Patched vector vertex shader")
