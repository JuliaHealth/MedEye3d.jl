with open("src/display/Vulkan/VulkanShaders.jl", "r") as f:
    code = f.read()

# Add PushConstants to fragment shader
pc_block = """
    layout(push_constant) uniform PushConstants {
        vec2 uvScale;
        vec2 uvOffset;
        vec2 ndcMin;
        vec2 ndcMax;
        vec2 crosshairUV;
        int showCrosshair;
        int padding;
    } pc;
"""
# find "layout(set = 1, binding = 0) uniform Settings" and insert before it
code = code.replace("layout(set = 1, binding = 0) uniform Settings", pc_block + "\n    layout(set = 1, binding = 0) uniform Settings")

# At the end of fragment shader, just before FragColor = vec4(baseColor, 1.0);
crosshair_code = """
        if (pc.showCrosshair == 1) {
            float dx = abs(TexCoord0.x - pc.crosshairUV.x);
            float dy = abs(TexCoord0.y - pc.crosshairUV.y);
            // Draw a thin crosshair (e.g. < 0.002 in UV space)
            if ((dx < 0.002 && dy < 0.1) || (dy < 0.002 && dx < 0.1)) {
                // simple invert or bright green
                baseColor = vec3(0.0, 1.0, 0.0);
            }
        }
        FragColor = vec4(baseColor, 1.0);
"""
code = code.replace("FragColor = vec4(baseColor, 1.0);", crosshair_code)

with open("src/display/Vulkan/VulkanShaders.jl", "w") as f:
    f.write(code)
