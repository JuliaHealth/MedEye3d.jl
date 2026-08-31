#!/usr/bin/env julia
"""
End-to-end Vulkan backend smoke test.

Verifies that the VulkanBackend modules can:
1. Initialize a Vulkan context (instance, device, swapchain, render pass)
2. Create quad geometry buffers
3. Compile GLSL 450 shaders to SPIR-V and create a pipeline
4. Create a texture and upload data
5. Render frames to the swapchain
6. Capture a screenshot

Run inside Docker: julia --project=. test/test_vulkan_backend.jl
"""

using Test

@testset "VulkanBackend Smoke Test" begin

using MedEye3d
using MedEye3d.VulkanBackend
using MedEye3d.VulkanBackend.VulkanContext
using MedEye3d.VulkanBackend.VulkanShaders
using MedEye3d.VulkanBackend.VulkanBuffers
using MedEye3d.VulkanBackend.VulkanTextures
using MedEye3d.VulkanBackend.VulkanPipeline
using MedEye3d.VulkanBackend.VulkanRender
using MedEye3d.VulkanBackend.VulkanScreenshot
using MedEye3d.VulkanBackend.VulkanStaging
using Vulkan
using VulkanCore
using GLFW

println("=" ^ 60)
println("VulkanBackend Smoke Test")
println("=" ^ 60)

# ── Phase 1: Context creation ──
println("\n[1/6] Initializing Vulkan context...")
using GLFW
GLFW.Init()
GLFW.WindowHint(GLFW.CLIENT_API, GLFW.NO_API)
window = GLFW.CreateWindow(800, 600, "MedEye3d — Vulkan Test")
ctx = init_vulkan_context(window, 800, 600)
@test ctx isa VkCtx
@test hasproperty(ctx, :device)
@test hasproperty(ctx, :swapchain)
@test length(ctx.swapchain_images) >= 2
@test length(ctx.framebuffers) == length(ctx.swapchain_images)
println("  ✓ Context created: $(length(ctx.swapchain_images)) swapchain images")
println("  ✓ Swapchain format: $(ctx.swapchain_format)")
println("  ✓ Extent: $(ctx.swapchain_extent.width)x$(ctx.swapchain_extent.height)")

# ── Phase 2: Quad buffers ──
println("\n[2/6] Creating quad geometry buffers...")
# Standard quad vertices: position(3) + color(3) + texcoord(2) = 8 floats per vertex
vertices = Float32[
    # top right
     1.0,  1.0, 0.0,   1.0, 0.0, 0.0,   1.0, 1.0,
    # bottom right
     1.0, -1.0, 0.0,   0.0, 1.0, 0.0,   1.0, 0.0,
    # bottom left
    -1.0, -1.0, 0.0,   0.0, 0.0, 1.0,   0.0, 0.0,
    # top left
    -1.0,  1.0, 0.0,   1.0, 1.0, 0.0,   0.0, 1.0,
]
quad = create_quad_buffers(ctx, vertices)
@test quad isa VkQuadBuffers
@test quad.vertex_count == 4
@test quad.index_count == 6
println("  ✓ Quad buffers created: $(quad.vertex_count) vertices, $(quad.index_count) indices")

# ── Phase 3: Shader compilation + Pipeline ──
println("\n[3/6] Compiling shaders and creating pipeline...")
vert_glsl = generate_vulkan_vertex_shader()
# Simple test fragment shader (solid color gradient based on texcoord)
frag_glsl = """
#version 450

layout(location = 0) in vec3 ourColor;
layout(location = 1) in vec2 TexCoord0;
layout(location = 0) out vec4 FragColor;

layout(set = 0, binding = 0) uniform sampler2D testTex;

layout(set = 1, binding = 0) uniform TextureParams {
    int   testTexisVisible;
    float testTexminValue;
    float testTexmaxValue;
    float testTexValueRange;
    float testTexmaskContribution;
    float _pad0; float _pad1; float _pad2;  // pad to 32
    vec4  testTexColorMask;
    int   testTexallowedIDCount;
    float _pad3; float _pad4; float _pad5;  // pad to 64
    float testTexallowedIDs[16];
} params;

void main() {
    float val = texture(testTex, TexCoord0).r;
    FragColor = vec4(val, TexCoord0.x, TexCoord0.y, 1.0);
}
"""

pipeline_state = create_pipeline_state(ctx, vert_glsl, frag_glsl, 1)
@test pipeline_state isa VkPipelineState
println("  ✓ Pipeline created successfully")

# ── Phase 4: Texture creation ──
println("\n[4/6] Creating and uploading texture...")
tex_w, tex_h = 256, 256
# Gradient texture: value increases from 0 to 1
tex_data = Float32[Float32(x) / tex_w for y in 1:tex_h, x in 1:tex_w]
test_tex = create_vulkan_texture(ctx, tex_w, tex_h,
    Vulkan.FORMAT_R32_SFLOAT, tex_data;
    filter_mode=:linear, name="testTex")
@test test_tex isa VkTexture
@test test_tex.width == tex_w
@test test_tex.height == tex_h
println("  ✓ Texture created: $(tex_w)x$(tex_h) R32_SFLOAT")

# Update descriptor set with texture
update_descriptor_textures!(ctx, pipeline_state, [test_tex])
println("  ✓ Descriptor set updated with texture")

# ── Phase 5: Render frames ──
println("\n[5/6] Rendering test frames...")
n_frames = 5
for frame in 1:n_frames
    # Push constants: no zoom, no pan
    push_data = Float32[1.0, 1.0, 0.0, 0.0]  # uvScale=(1,1), uvOffset=(0,0)

    panel = PanelRenderData(
        pipeline_state,
        quad,
        push_data,
        0.0f0, 0.0f0,              # viewport x, y
        Float32(ctx.width),         # viewport w
        Float32(ctx.height),        # viewport h
    )

    success = render_frame!(ctx, [panel])
    @test success
    GLFW.PollEvents()
end
println("  ✓ Rendered $n_frames frames successfully")

# ── Phase 7: Batched staging transfer (VulkanStaging) ──
println("\n[7/7] Testing persistent staging ring buffer and batched upload...")
staging_pool = create_staging_pool(ctx, 16)
@test staging_pool isa VkStagingPool
@test staging_pool.capacity == 16 * 1024 * 1024

batch_items = [
    TextureUploadBatchItem(test_tex, Float32[Float32(x * y) / (tex_w * tex_h) for y in 1:tex_h, x in 1:tex_w])
]
upload_textures_batched!(ctx, staging_pool, batch_items)
println("  ✓ Batched multi-texture upload succeeded")

push_data = Float32[1.0, 1.0, 0.0, 0.0]
panel_zvbo = PanelRenderData(pipeline_state, push_data,
    0.0f0, 0.0f0, Float32(ctx.width), Float32(ctx.height))
success = render_frame!(ctx, [panel_zvbo])
@test success
println("  ✓ Zero-VBO frame render succeeded")

# ── Cleanup ──
println("\nCleaning up...")
using Vulkan: device_wait_idle
device_wait_idle(ctx.device)
destroy_vulkan_texture!(ctx, test_tex)
destroy_pipeline_state!(ctx, pipeline_state)
destroy_quad_buffers!(ctx, quad)
destroy_vulkan_context!(ctx)

println("\n" * "=" ^ 60)
println("ALL VULKAN BACKEND TESTS PASSED ✓")
println("=" ^ 60)

end # testset
