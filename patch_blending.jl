path = "src/display/Vulkan/VulkanPipeline.jl"
content = read(path, String)

old_code = """
    cba = [VulkanCore.VkPipelineColorBlendAttachmentState(
        UInt32(0), # blendEnable
        VulkanCore.VK_BLEND_FACTOR_SRC_ALPHA,
        VulkanCore.VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
        VulkanCore.VK_BLEND_OP_ADD,
        VulkanCore.VK_BLEND_FACTOR_ONE,
        VulkanCore.VK_BLEND_FACTOR_ZERO,
        VulkanCore.VK_BLEND_OP_ADD,
        VulkanCore.VK_COLOR_COMPONENT_R_BIT | VulkanCore.VK_COLOR_COMPONENT_G_BIT |
        VulkanCore.VK_COLOR_COMPONENT_B_BIT | VulkanCore.VK_COLOR_COMPONENT_A_BIT
    )]
"""

new_code = """
    cba = [VulkanCore.VkPipelineColorBlendAttachmentState(
        UInt32(1), # blendEnable (TRUE)
        VulkanCore.VK_BLEND_FACTOR_SRC_ALPHA,
        VulkanCore.VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
        VulkanCore.VK_BLEND_OP_ADD,
        VulkanCore.VK_BLEND_FACTOR_ONE,
        VulkanCore.VK_BLEND_FACTOR_ZERO,
        VulkanCore.VK_BLEND_OP_ADD,
        VulkanCore.VK_COLOR_COMPONENT_R_BIT | VulkanCore.VK_COLOR_COMPONENT_G_BIT |
        VulkanCore.VK_COLOR_COMPONENT_B_BIT | VulkanCore.VK_COLOR_COMPONENT_A_BIT
    )]
"""

content = replace(content, old_code => new_code)
write(path, content)
