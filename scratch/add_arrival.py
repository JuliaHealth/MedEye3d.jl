import re

with open("src/display/Vulkan/VulkanHeatDiffusion.jl", "r") as f:
    code = f.read()

# Add to struct
code = re.sub(r'(img_mask::Image.*)', r'\1\n    img_arrival::Image', code)
code = re.sub(r'(view_mask::ImageView)', r'\1\n    view_arrival::ImageView', code)
code = re.sub(r'(mem_mask::DeviceMemory)', r'\1\n    mem_arrival::DeviceMemory', code)

# Add to create_heat_diffusion_state
code = re.sub(r'(state.img_mask, state.mem_mask, state.view_mask = create_storage_image_3d\(dev, pdev, w, h, d\))',
              r'\1\n    state.img_arrival, state.mem_arrival, state.view_arrival = create_storage_image_3d(dev, pdev, w, h, d)', code)

# Clear in seed_only!
# First the image barrier for arrival
barrier_arrival = """    push!(barriers_init, ImageMemoryBarrier(
        C_NULL, ACCESS_TRANSFER_WRITE_BIT,
        IMAGE_LAYOUT_UNDEFINED, IMAGE_LAYOUT_GENERAL,
        0, 0, state.img_arrival, ImageSubresourceRange(IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1)))
"""
code = code.replace("    cmd_pipeline_barrier(cmd, MemoryBarrier[], BufferMemoryBarrier[], barriers_init;",
                    barrier_arrival + "    cmd_pipeline_barrier(cmd, MemoryBarrier[], BufferMemoryBarrier[], barriers_init;")

# Clear color
clear_arrival = """    cmd_clear_color_image(cmd, state.img_arrival, IMAGE_LAYOUT_GENERAL, ClearColorValue((1000000.0f0, 0.0f0, 0.0f0, 0.0f0)), [subres])
"""
code = code.replace("    cmd_clear_color_image(cmd, state.img_mask, IMAGE_LAYOUT_GENERAL, clear_color, [subres])",
                    "    cmd_clear_color_image(cmd, state.img_mask, IMAGE_LAYOUT_GENERAL, clear_color, [subres])\n" + clear_arrival)

# Also seed_init.comp must set arrival to 0. Wait! I can just write to img_arrival in diffuse_step instead!
# diffuse_step.comp needs to bind img_arrival.

with open("src/display/Vulkan/VulkanHeatDiffusion.jl", "w") as f:
    f.write(code)

print("Added img_arrival to VulkanHeatDiffusion.jl")
