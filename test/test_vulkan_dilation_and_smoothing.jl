#!/usr/bin/env julia
"""
End-to-end GPU tests for the two custom Vulkan shader extensions to Heat-GDT:
1. 3D GPU morphological dilation (`dilate_mask.comp`, radii 0, 1, 2, 3) covering strong lesion edges
   plus live `redilate_current_mask!` and `apply_box_mask_to_segvol!`.
2. Automatic strong PET edge detection (`detect_strong_pet_lesion_edges`).
3. 3-pass separable GPU Gaussian edge smoothing & gap bridging (`gaussian_smooth.comp`) and
   exact bit-for-bit reversibility back to `sigma = 0.0` without mutating CPU arrays or HDF5.
"""

using Test
using MedEye3d
using MedEye3d.VulkanBackend
using MedEye3d.VulkanBackend.VulkanContext
using MedEye3d.VulkanBackend.VulkanHeatDiffusion
using GLFW

println("=" ^ 70)
println("  Vulkan Heat-GDT Dilation & Reversible Gaussian Smoothing Test")
println("=" ^ 70)

GLFW.Init()
GLFW.WindowHint(GLFW.CLIENT_API, GLFW.NO_API)
GLFW.WindowHint(GLFW.VISIBLE, 0)
window = GLFW.CreateWindow(256, 256, "Vulkan Dilation & Smoothing Test")
ctx = init_vulkan_context(window, 256, 256)

try
    @testset "Vulkan Heat-GDT Dilation & Reversible Gaussian Smoothing" begin
        w, h, d = 64, 64, 64
        cx, cy, cz = 32, 32, 32
        strong_edge_D = Float32(exp(-10.095886 * 0.85))  # E = 0.85 strong barrier

        # ── 1 & 2. 3D GPU Dilation (0..3 vox) + Auto Strong PET Edge Detection ──
        @testset "3D GPU Dilation (0..3) & Strong PET Edge Detection" begin
            state = HeatDiffusionState()
            init_heat_diffusion!(state, ctx, w, h, d)
            @test state.is_initialized

            D = ones(Float32, w, h, d)
            shell_coords = CartesianIndex{3}[]
            for z in 1:d, y in 1:h, x in 1:w
                r = sqrt((x - cx)^2 + (y - cy)^2 + (z - cz)^2)
                if 8.5 <= r <= 10.5
                    D[x, y, z] = strong_edge_D
                    if 8.5 <= r <= 10.0
                        push!(shell_coords, CartesianIndex(x, y, z))
                    end
                end
            end

            upload_diffusivity!(state, ctx, D)
            seed_only!(state, ctx, cx, cy, cz; seed_radius=1)
            run_incremental_steps!(state, ctx, 40; theta=0.001f0, copy_box_to_cpu=true, dilation_radius=0)

            mask0 = read_mask_to_cpu(state, ctx)
            v0 = count(mask0 .> 0.5f0)
            shell_cov0 = count(idx -> mask0[idx] > 0.5f0, shell_coords) / length(shell_coords)
            println("  Dilation=0 vox: count=$v0, shell_coverage=$(round(shell_cov0 * 100, digits=1))%")
            @test v0 > 0
            @test mask0[cx, cy, cz] > 0.5f0
            @test shell_cov0 < 0.75

            # Check PET edge detection using un-dilated staging box
            pet_strong = ones(Float32, w, h, d)
            for z in 1:d, y in 1:h, x in 1:w
                r = sqrt((x - cx)^2 + (y - cy)^2 + (z - cz)^2)
                if r <= 9.0
                    pet_strong[x, y, z] = 10.0f0
                end
            end
            @test detect_strong_pet_lesion_edges(state, D, pet_strong, cx, cy, cz, state.current_edge_params) == true

            pet_flat = ones(Float32, w, h, d)
            @test detect_strong_pet_lesion_edges(state, D, pet_flat, cx, cy, cz, state.current_edge_params) == false

            D_weak = fill(0.85f0, w, h, d)
            @test detect_strong_pet_lesion_edges(state, D_weak, pet_strong, cx, cy, cz, (0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0)) == false

            # Redilate to 1, 2, 3 and verify strictly monotonic growth + strong edge shell coverage at r=2
            counts = Int[v0]
            seg_vol = zeros(UInt16, w, h, d)
            for rad in 1:3
                redilate_current_mask!(state, ctx, rad; theta=0.001f0, copy_box_to_cpu=true)
                mask_r = read_mask_to_cpu(state, ctx)
                vr = count(mask_r .> 0.5f0)
                push!(counts, vr)

                fill!(seg_vol, UInt16(0))
                box_vr = VulkanHeatDiffusion.apply_box_mask_to_segvol!(state, seg_vol, UInt16(5); theta=0.001f0)
                @test box_vr == vr
                @test count(seg_vol .== UInt16(5)) == vr

                shell_cov_r = count(idx -> mask_r[idx] > 0.5f0, shell_coords) / length(shell_coords)
                println("  Dilation=$rad vox: count=$vr, shell_coverage=$(round(shell_cov_r * 100, digits=1))%")
                if rad == 2
                    @test shell_cov_r > 0.90
                end
            end
            @test counts[1] < counts[2] < counts[3] < counts[4]

            # Revert dilation back to 0 and confirm exact count v0
            redilate_current_mask!(state, ctx, 0; theta=0.001f0, copy_box_to_cpu=true)
            mask0_back = read_mask_to_cpu(state, ctx)
            @test count(mask0_back .> 0.5f0) == v0

            destroy_heat_diffusion!(state, ctx)
        end

        # ── 3. Reversible 3D Separable Gaussian Smoothing & Gap Bridging ──
        @testset "3-Pass GPU Gaussian Smoothing & Bit-for-Bit Reversibility" begin
            state = HeatDiffusionState()
            init_heat_diffusion!(state, ctx, w, h, d)

            D = fill(0.95f0, w, h, d)
            for z in 1:d, y in 1:h, x in 1:w
                r = sqrt((x - cx)^2 + (y - cy)^2 + (z - cz)^2)
                if 7.5 <= r <= 9.5
                    D[x, y, z] = strong_edge_D
                end
            end
            # Puncture a 2-voxel wide gap through the barrier at +X side
            D[38:43, 32:33, 32:33] .= 0.95f0
            D_cpu_backup = copy(D)

            upload_diffusivity!(state, ctx, D)

            # A. Unsmoothed (sigma = 0.0): expect heat leak into x > 42
            apply_gaussian_smoothing!(state, ctx, 0.0f0)
            seed_only!(state, ctx, cx, cy, cz; seed_radius=1)
            run_incremental_steps!(state, ctx, 60; theta=0.001f0, copy_box_to_cpu=true, dilation_radius=0)
            mask_unsmoothed = read_mask_to_cpu(state, ctx)
            vox_unsmoothed = count(mask_unsmoothed .> 0.5f0)
            ext_unsmoothed = count(@view(mask_unsmoothed[43:w, :, :]) .> 0.5f0)
            println("  Unsmoothed (sigma=0.0): total=$vox_unsmoothed, exterior(x>=43)=$ext_unsmoothed")
            @test ext_unsmoothed > 0

            # B. Smoothed (sigma = 1.5): closes 2-voxel gap without mutating CPU array
            apply_gaussian_smoothing!(state, ctx, 1.5f0)
            @test D == D_cpu_backup
            seed_only!(state, ctx, cx, cy, cz; seed_radius=1)
            run_incremental_steps!(state, ctx, 60; theta=0.001f0, copy_box_to_cpu=true, dilation_radius=0)
            mask_smoothed = read_mask_to_cpu(state, ctx)
            vox_smoothed = count(mask_smoothed .> 0.5f0)
            ext_smoothed = count(@view(mask_smoothed[43:w, :, :]) .> 0.5f0)
            println("  Smoothed   (sigma=1.5): total=$vox_smoothed, exterior(x>=43)=$ext_smoothed")
            @test vox_smoothed < vox_unsmoothed
            @test ext_smoothed < ext_unsmoothed

            # C. Reversed (sigma -> 0.0): restores img_diffusivity_raw bit-for-bit
            apply_gaussian_smoothing!(state, ctx, 0.0f0)
            seed_only!(state, ctx, cx, cy, cz; seed_radius=1)
            run_incremental_steps!(state, ctx, 60; theta=0.001f0, copy_box_to_cpu=true, dilation_radius=0)
            mask_reverted = read_mask_to_cpu(state, ctx)
            vox_reverted = count(mask_reverted .> 0.5f0)
            println("  Reverted   (sigma=0.0): total=$vox_reverted (exact match=$(mask_reverted == mask_unsmoothed))")
            @test vox_reverted == vox_unsmoothed
            @test mask_reverted == mask_unsmoothed
            @test D == D_cpu_backup

            destroy_heat_diffusion!(state, ctx)
        end
    end
finally
    GLFW.DestroyWindow(window)
    GLFW.Terminate()
end
