import re

with open("src/display/Vulkan/VulkanHeatDiffusion.jl", "r") as f:
    code = f.read()

code = code.replace(
    """    box_max_y::Int32
    box_max_z::Int32
    box_max_w::Int32
end""",
    """    box_max_y::Int32
    box_max_z::Int32
    box_max_w::Int32
    theta::Float32
    current_step::UInt32
end"""
)

with open("src/display/Vulkan/VulkanHeatDiffusion.jl", "w") as f:
    f.write(code)

