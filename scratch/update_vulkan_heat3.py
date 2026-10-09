import re

with open("src/display/Vulkan/VulkanHeatDiffusion.jl", "r") as f:
    code = f.read()

code = code.replace(
    """    # slice to texture: u_ping/u_pong → tex_out
    push!(writes, WriteDescriptorSet(state.ds_slice_ping, 0, 0, DESCRIPTOR_TYPE_STORAGE_IMAGE, [di(state.view_u_ping)], [], []))
    push!(writes, WriteDescriptorSet(state.ds_slice_pong, 0, 0, DESCRIPTOR_TYPE_STORAGE_IMAGE, [di(state.view_u_pong)], [], []))""",
    """    # slice to texture: arrival → tex_out
    push!(writes, WriteDescriptorSet(state.ds_slice_ping, 0, 0, DESCRIPTOR_TYPE_STORAGE_IMAGE, [di(state.view_arrival)], [], []))
    push!(writes, WriteDescriptorSet(state.ds_slice_pong, 0, 0, DESCRIPTOR_TYPE_STORAGE_IMAGE, [di(state.view_arrival)], [], []))"""
)

with open("src/display/Vulkan/VulkanHeatDiffusion.jl", "w") as f:
    f.write(code)

