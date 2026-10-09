import re

with open("src/display/Vulkan/VulkanHeatDiffusion.jl", "r") as f:
    code = f.read()

# Update ds_diffuse_ping_to_pong
code = code.replace(
    """    # diffuse: ping→pong (D, ping_in, pong_out)
    push!(writes, WriteDescriptorSet(state.ds_diffuse_ping_to_pong, 0, 0, DESCRIPTOR_TYPE_STORAGE_IMAGE, [di(state.view_diffusivity)], [], []))
    push!(writes, WriteDescriptorSet(state.ds_diffuse_ping_to_pong, 1, 0, DESCRIPTOR_TYPE_STORAGE_IMAGE, [di(state.view_u_ping)], [], []))
    push!(writes, WriteDescriptorSet(state.ds_diffuse_ping_to_pong, 2, 0, DESCRIPTOR_TYPE_STORAGE_IMAGE, [di(state.view_u_pong)], [], []))""",
    """    # diffuse: ping→pong (D, ping_in, pong_out, arrival_out)
    push!(writes, WriteDescriptorSet(state.ds_diffuse_ping_to_pong, 0, 0, DESCRIPTOR_TYPE_STORAGE_IMAGE, [di(state.view_diffusivity)], [], []))
    push!(writes, WriteDescriptorSet(state.ds_diffuse_ping_to_pong, 1, 0, DESCRIPTOR_TYPE_STORAGE_IMAGE, [di(state.view_u_ping)], [], []))
    push!(writes, WriteDescriptorSet(state.ds_diffuse_ping_to_pong, 2, 0, DESCRIPTOR_TYPE_STORAGE_IMAGE, [di(state.view_u_pong)], [], []))
    push!(writes, WriteDescriptorSet(state.ds_diffuse_ping_to_pong, 3, 0, DESCRIPTOR_TYPE_STORAGE_IMAGE, [di(state.view_arrival)], [], []))"""
)

# Update ds_diffuse_pong_to_ping
code = code.replace(
    """    # diffuse: pong→ping (D, pong_in, ping_out)
    push!(writes, WriteDescriptorSet(state.ds_diffuse_pong_to_ping, 0, 0, DESCRIPTOR_TYPE_STORAGE_IMAGE, [di(state.view_diffusivity)], [], []))
    push!(writes, WriteDescriptorSet(state.ds_diffuse_pong_to_ping, 1, 0, DESCRIPTOR_TYPE_STORAGE_IMAGE, [di(state.view_u_pong)], [], []))
    push!(writes, WriteDescriptorSet(state.ds_diffuse_pong_to_ping, 2, 0, DESCRIPTOR_TYPE_STORAGE_IMAGE, [di(state.view_u_ping)], [], []))""",
    """    # diffuse: pong→ping (D, pong_in, ping_out, arrival_out)
    push!(writes, WriteDescriptorSet(state.ds_diffuse_pong_to_ping, 0, 0, DESCRIPTOR_TYPE_STORAGE_IMAGE, [di(state.view_diffusivity)], [], []))
    push!(writes, WriteDescriptorSet(state.ds_diffuse_pong_to_ping, 1, 0, DESCRIPTOR_TYPE_STORAGE_IMAGE, [di(state.view_u_pong)], [], []))
    push!(writes, WriteDescriptorSet(state.ds_diffuse_pong_to_ping, 2, 0, DESCRIPTOR_TYPE_STORAGE_IMAGE, [di(state.view_u_ping)], [], []))
    push!(writes, WriteDescriptorSet(state.ds_diffuse_pong_to_ping, 3, 0, DESCRIPTOR_TYPE_STORAGE_IMAGE, [di(state.view_arrival)], [], []))"""
)

# Update slice_heat_to_texture.comp binding in ds_slice_ping and ds_slice_pong
# Wait! How is ds_slice_ping updated?
