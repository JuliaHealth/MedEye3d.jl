using Vulkan
b = ImageMemoryBarrier(
        C_NULL, ACCESS_SHADER_READ_BIT, ACCESS_SHADER_WRITE_BIT,
        IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, IMAGE_LAYOUT_GENERAL,
        0, 0, C_NULL, ImageSubresourceRange(IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1)
    )
println("Success!")
