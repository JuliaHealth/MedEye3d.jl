import re

with open("src/display/Vulkan/VulkanHeatDiffusion.jl", "r") as f:
    code = f.read()

code = code.replace('"slice_heat_to_texture.spv"', '"slice_heat_to_texture_v2.spv"')

code = code.replace(
    """    # diffuse_step: 3 storage images (D readonly, u_in readonly, u_out writeonly)
    bindings_diffuse = [
        DescriptorSetLayoutBinding(0, DESCRIPTOR_TYPE_STORAGE_IMAGE, SHADER_STAGE_COMPUTE_BIT; descriptor_count=1),
        DescriptorSetLayoutBinding(1, DESCRIPTOR_TYPE_STORAGE_IMAGE, SHADER_STAGE_COMPUTE_BIT; descriptor_count=1),
        DescriptorSetLayoutBinding(2, DESCRIPTOR_TYPE_STORAGE_IMAGE, SHADER_STAGE_COMPUTE_BIT; descriptor_count=1)
    ]""",
    """    # diffuse_step: 4 storage images (D readonly, u_in readonly, u_out writeonly, arrival writeonly)
    bindings_diffuse = [
        DescriptorSetLayoutBinding(0, DESCRIPTOR_TYPE_STORAGE_IMAGE, SHADER_STAGE_COMPUTE_BIT; descriptor_count=1),
        DescriptorSetLayoutBinding(1, DESCRIPTOR_TYPE_STORAGE_IMAGE, SHADER_STAGE_COMPUTE_BIT; descriptor_count=1),
        DescriptorSetLayoutBinding(2, DESCRIPTOR_TYPE_STORAGE_IMAGE, SHADER_STAGE_COMPUTE_BIT; descriptor_count=1),
        DescriptorSetLayoutBinding(3, DESCRIPTOR_TYPE_STORAGE_IMAGE, SHADER_STAGE_COMPUTE_BIT; descriptor_count=1)
    ]"""
)

code = code.replace(
    "pc_diffuse = [PushConstantRange(SHADER_STAGE_COMPUTE_BIT, 0, 48)]",
    "pc_diffuse = [PushConstantRange(SHADER_STAGE_COMPUTE_BIT, 0, 64)]"
)

# slice_heat_to_texture push constant size
code = code.replace(
    "pc_slice = [PushConstantRange(SHADER_STAGE_COMPUTE_BIT, 0, 16)]",
    "pc_slice = [PushConstantRange(SHADER_STAGE_COMPUTE_BIT, 0, 20)]"
)

with open("src/display/Vulkan/VulkanHeatDiffusion.jl", "w") as f:
    f.write(code)

print("Updated VulkanHeatDiffusion.jl")
