#!/usr/bin/env julia
# Minimal test to verify the Heat-GDT GPU pipeline works independently
# Run: julia --project=/mnt/big/project_ssd/project_ssd/MedEye3d.jl scratch/test_heatgdt_pipeline.jl

println("=== Heat-GDT Pipeline Test ===")

# Activate project
using Pkg
Pkg.activate("/mnt/big/project_ssd/project_ssd/MedEye3d.jl")

println("Loading MedEye3d...")
using MedEye3d
using MedEye3d.SegmentationDisplay.MakieEventHandlers.VulkanHeatDiffusion

println("Loading Vulkan...")
using Vulkan

# Check if we can create a minimal Vulkan context
println("\n--- This test verifies the VulkanHeatDiffusion module loads correctly ---")
println("Module loaded: VulkanHeatDiffusion")
println("Exports: $(names(VulkanHeatDiffusion))")

# Check struct sizes match shader push constants
println("\nDiffuseStepPC size: $(sizeof(VulkanHeatDiffusion.DiffuseStepPC)) bytes (expected: 48)")
println("ExtractMaskPC size: $(sizeof(VulkanHeatDiffusion.ExtractMaskPC)) bytes (expected: 48)")

# Verify shader files exist
shader_dir = joinpath(dirname(dirname(pathof(MedEye3d))), "src", "display", "Vulkan", "shaders")
println("\nShader directory: $shader_dir")
for spv in ["seed_init.spv", "diffuse_step.spv", "extract_mask.spv", "slice_heat_to_texture.spv"]
    path = joinpath(shader_dir, spv)
    if isfile(path)
        println("  ✓ $spv ($(filesize(path)) bytes)")
    else
        println("  ✗ $spv MISSING!")
    end
end

# Check for v2 shader (should NOT exist)
v2_path = joinpath(shader_dir, "slice_heat_to_texture_v2.spv")
if isfile(v2_path)
    println("  ⚠ slice_heat_to_texture_v2.spv EXISTS (should be removed!)")
else
    println("  ✓ slice_heat_to_texture_v2.spv correctly absent")
end

println("\n=== Test Complete ===")
