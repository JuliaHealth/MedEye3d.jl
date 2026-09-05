"""
    VulkanBackend

Top-level module that exports the complete Vulkan rendering backend for
MedEye3d. Includes context management, shader compilation, pipeline
creation, texture management, buffer management, frame rendering, and
screenshot capture.

## Usage

```julia
using .VulkanBackend

# Initialize
ctx = VulkanContext.init_vulkan_context(1200, 900)

# Create textures, pipeline, buffers...
# Render frames...
# Capture screenshots...

# Cleanup
VulkanContext.destroy_vulkan_context!(ctx)
```

## Module Structure

- `VulkanContext` — Instance, device, swapchain, render pass, framebuffers
- `VulkanShaders` — GLSL 450 → SPIR-V compilation and shader generation
- `VulkanBuffers` — Vertex and index buffer management
- `VulkanTextures` — Texture image creation, staging uploads, samplers
- `VulkanPipeline` — Descriptor sets, pipeline layout, graphics pipeline, UBO
- `VulkanRender` — Frame recording, submission, presentation
- `VulkanScreenshot` — Swapchain image readback to PNG
"""
module VulkanBackend

# Sub-modules are included in dependency order
include("VulkanContext.jl")
include("VulkanShaders.jl")
include("VulkanBuffers.jl")
include("VulkanTextures.jl")
include("VulkanPipeline.jl")
include("VulkanRender.jl")
include("VulkanScreenshot.jl")
include("VulkanStaging.jl")

# Re-export all sub-modules
using .VulkanContext
using .VulkanShaders
using .VulkanBuffers
using .VulkanTextures
using .VulkanPipeline
using .VulkanRender
using .VulkanScreenshot
using .VulkanStaging

export VulkanContext, VulkanShaders, VulkanBuffers, VulkanTextures
export VulkanPipeline, VulkanRender, VulkanScreenshot, VulkanStaging

end # module VulkanBackend
