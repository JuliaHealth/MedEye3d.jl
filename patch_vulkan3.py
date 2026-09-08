with open("src/display/Vulkan/VulkanShaders.jl", "r") as f:
    code = f.read()

old_pc = """    layout(push_constant) uniform PushConstants {
        vec2 uvScale;
        vec2 uvOffset;
        vec2 ndcMin;   // unused by VBO shader, but must match layout
        vec2 ndcMax;
    } pc;"""
new_pc = """    layout(push_constant) uniform PushConstants {
        vec2 uvScale;
        vec2 uvOffset;
        vec2 ndcMin;
        vec2 ndcMax;
        vec2 crosshairUV;
        int showCrosshair;
        int padding;
    } pc;"""
code = code.replace(old_pc, new_pc)

old_pc2 = """    layout(push_constant) uniform PushConstants {
        vec2 uvScale;
        vec2 uvOffset;
        vec2 ndcMin;   // (left, bottom) in NDC
        vec2 ndcMax;   // (right, top) in NDC
    } pc;"""
code = code.replace(old_pc2, new_pc)

with open("src/display/Vulkan/VulkanShaders.jl", "w") as f:
    f.write(code)
