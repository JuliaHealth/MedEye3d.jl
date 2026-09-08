import re

with open("src/display/Vulkan/VulkanPipeline.jl", "r") as f:
    code = f.read()

code = code.replace("VulkanCore.VK_SHADER_STAGE_VERTEX_BIT", "VulkanCore.VK_SHADER_STAGE_VERTEX_BIT | VulkanCore.VK_SHADER_STAGE_FRAGMENT_BIT")

with open("src/display/Vulkan/VulkanPipeline.jl", "w") as f:
    f.write(code)

with open("src/display/Vulkan/VulkanRender.jl", "r") as f:
    code = f.read()

code = code.replace("SHADER_STAGE_VERTEX_BIT", "SHADER_STAGE_VERTEX_BIT | SHADER_STAGE_FRAGMENT_BIT")

with open("src/display/Vulkan/VulkanRender.jl", "w") as f:
    f.write(code)
