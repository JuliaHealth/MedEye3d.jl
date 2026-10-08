#!/usr/bin/env julia
"""
Test that the Heat-GDT label saving pipeline works correctly:
1. extract_patch → 64³ diffusivity patch
2. run_heatgdt → 64³ binary mask
3. insert_patch! → write back into full volume at correct coordinates
4. Verify: correct voxels are labeled, no out-of-bounds writes

This test does NOT require a GPU or running GUI — it tests the pure data flow.
"""

using Test

println("=" ^ 70)
println("  Heat-GDT Label Save Roundtrip Test")
println("=" ^ 70)

# ─── Replicate-pad helper (matches InferenceClient.jl) ─────────────────
function _replicate_pad(arr)
    s = size(arr)
    padded = similar(arr, s[1]+2, s[2]+2, s[3]+2, s[4])
    padded[2:end-1, 2:end-1, 2:end-1, :] .= arr
    padded[1, :, :, :] .= padded[2, :, :, :]
    padded[end, :, :, :] .= padded[end-1, :, :, :]
    padded[:, 1, :, :] .= padded[:, 2, :, :]
    padded[:, end, :, :] .= padded[:, end-1, :, :]
    padded[:, :, 1, :] .= padded[:, :, 2, :]
    padded[:, :, end, :] .= padded[:, :, end-1, :]
    return padded
end

# ─── Minimal extract_patch (same logic as InferenceClient.jl) ──────────
function extract_patch(vol::Array{Float32, 3}, cx::Int, cy::Int, cz::Int;
                       patch_size::Int=64, pad_val::Float32=0.0f0)
    w, h, d = size(vol)
    patch = fill(pad_val, patch_size, patch_size, patch_size)
    hw = patch_size ÷ 2
    
    src_x1 = max(1, cx - hw)
    src_x2 = min(w, cx + (patch_size - hw - 1))
    dst_x1 = 1 + (src_x1 - (cx - hw))
    dst_x2 = patch_size - ((cx + (patch_size - hw - 1)) - src_x2)

    src_y1 = max(1, cy - hw)
    src_y2 = min(h, cy + (patch_size - hw - 1))
    dst_y1 = 1 + (src_y1 - (cy - hw))
    dst_y2 = patch_size - ((cy + (patch_size - hw - 1)) - src_y2)

    src_z1 = max(1, cz - hw)
    src_z2 = min(d, cz + (patch_size - hw - 1))
    dst_z1 = 1 + (src_z1 - (cz - hw))
    dst_z2 = patch_size - ((cz + (patch_size - hw - 1)) - src_z2)
    
    patch[dst_x1:dst_x2, dst_y1:dst_y2, dst_z1:dst_z2] .= vol[src_x1:src_x2, src_y1:src_y2, src_z1:src_z2]
    return patch
end

# ─── Minimal insert_patch! (same logic as InferenceClient.jl) ──────────
function insert_patch!(vol::AbstractArray{T, 3}, patch::AbstractArray{<:Real, 3},
                       cx::Int, cy::Int, cz::Int; label_val::T=T(1)) where T
    w, h, d = size(vol)
    pw, ph, pd = size(patch)
    
    hw = pw ÷ 2
    hh = ph ÷ 2
    hd = pd ÷ 2
    
    src_x1 = max(1, cx - hw)
    src_x2 = min(w, cx + (pw - hw - 1))
    dst_x1 = 1 + (src_x1 - (cx - hw))
    dst_x2 = pw - ((cx + (pw - hw - 1)) - src_x2)

    src_y1 = max(1, cy - hh)
    src_y2 = min(h, cy + (ph - hh - 1))
    dst_y1 = 1 + (src_y1 - (cy - hh))
    dst_y2 = ph - ((cy + (ph - hh - 1)) - src_y2)

    src_z1 = max(1, cz - hd)
    src_z2 = min(d, cz + (pd - hd - 1))
    dst_z1 = 1 + (src_z1 - (cz - hd))
    dst_z2 = pd - ((cz + (pd - hd - 1)) - src_z2)
    
    mask_slice = patch[dst_x1:dst_x2, dst_y1:dst_y2, dst_z1:dst_z2]
    
    @views target_slice = vol[src_x1:src_x2, src_y1:src_y2, src_z1:src_z2]
    for i in eachindex(mask_slice)
        if mask_slice[i] > 0
            target_slice[i] = label_val
        end
    end
    vol[src_x1:src_x2, src_y1:src_y2, src_z1:src_z2] .= target_slice
end

# ─── Minimal CPU heat diffusion (replicate BC) ────────────────────────
function run_heatgdt_cpu(D_vol::Array{Float32, 3}, cx::Int, cy::Int, cz::Int;
                          K::Int=40, dt::Float32=0.16f0, theta::Float32=0.001f0, tau::Float32=0.0001f0)
    D_patch = extract_patch(D_vol, cx, cy, cz; pad_val=1.0f0)
    seed = zeros(Float32, 64, 64, 64)
    seed[33, 33, 33] = 1.0f0
    
    N = 64
    D_4d = reshape(D_patch, N, N, N, 1)
    u = reshape(seed, N, N, N, 1)
    
    D_padded = _replicate_pad(D_4d)
    D_xp_nbr = D_padded[3:N+2, 2:N+1, 2:N+1, :]
    D_xm_nbr = D_padded[1:N,   2:N+1, 2:N+1, :]
    D_yp_nbr = D_padded[2:N+1, 3:N+2, 2:N+1, :]
    D_ym_nbr = D_padded[2:N+1, 1:N,   2:N+1, :]
    D_zp_nbr = D_padded[2:N+1, 2:N+1, 3:N+2, :]
    D_zm_nbr = D_padded[2:N+1, 2:N+1, 1:N,   :]
    
    D_xp = 0.5f0 .* (D_4d .+ D_xp_nbr)
    D_xm = 0.5f0 .* (D_4d .+ D_xm_nbr)
    D_yp = 0.5f0 .* (D_4d .+ D_yp_nbr)
    D_ym = 0.5f0 .* (D_4d .+ D_ym_nbr)
    D_zp = 0.5f0 .* (D_4d .+ D_zp_nbr)
    D_zm = 0.5f0 .* (D_4d .+ D_zm_nbr)
    
    for k in 1:K
        u_padded = _replicate_pad(u)
        u_xp = u_padded[3:N+2, 2:N+1, 2:N+1, :]
        u_xm = u_padded[1:N,   2:N+1, 2:N+1, :]
        u_yp = u_padded[2:N+1, 3:N+2, 2:N+1, :]
        u_ym = u_padded[2:N+1, 1:N,   2:N+1, :]
        u_zp = u_padded[2:N+1, 2:N+1, 3:N+2, :]
        u_zm = u_padded[2:N+1, 2:N+1, 1:N,   :]
        
        flux = D_xp .* (u_xp .- u) .+
               D_xm .* (u_xm .- u) .+
               D_yp .* (u_yp .- u) .+
               D_ym .* (u_ym .- u) .+
               D_zp .* (u_zp .- u) .+
               D_zm .* (u_zm .- u)
        u = u .+ dt .* flux
    end
    
    mask_field = 1.0f0 ./ (1.0f0 .+ exp.(-(u .- theta) ./ tau))
    binary_mask = UInt8.(mask_field[:, :, :, 1] .> 0.5f0)
    return binary_mask
end

@testset "Label Save Roundtrip Tests" begin
    
    @testset "1. Basic patch extract → insert roundtrip" begin
        # Create a known 128³ volume
        vol = zeros(Float32, 128, 128, 128)
        vol[60:68, 60:68, 60:68] .= 1.0f0  # A small 9³ cube
        
        cx, cy, cz = 64, 64, 64
        patch = extract_patch(vol, cx, cy, cz; pad_val=0.0f0)
        
        @test size(patch) == (64, 64, 64)
        # The 9³ cube should be in the patch (centered around patch center)
        @test sum(patch .> 0) == 9^3
        println("  ✓ Extract patch: correct size and content")
        
        # Create a fake mask (same shape) and insert back
        fake_mask = zeros(UInt8, 64, 64, 64)
        fake_mask[30:36, 30:36, 30:36] .= 1  # 7³ cube near center
        
        label_vol = zeros(Float32, 128, 128, 128)
        insert_patch!(label_vol, fake_mask, cx, cy, cz; label_val=1.0f0)
        
        labeled_count = count(label_vol .> 0)
        @test labeled_count == 7^3  # 343 voxels
        println("  ✓ Insert patch roundtrip: $labeled_count voxels labeled (expected $(7^3))")
    end
    
    @testset "2. Insert patch at volume boundary (no out-of-bounds)" begin
        vol = zeros(Float32, 100, 100, 100)
        mask = ones(UInt8, 64, 64, 64)  # Full 64³ mask
        
        # Insert near corner — should clip correctly
        insert_patch!(vol, mask, 5, 5, 5; label_val=2.0f0)
        labeled = count(vol .> 0)
        
        # At (5,5,5), half-width=32: covers x=[5-32, 5+31]=[−27, 36] → clipped to [1, 36]
        expected_x = 36  # 1:36
        expected_y = 36
        expected_z = 36
        expected_total = expected_x * expected_y * expected_z
        
        @test labeled == expected_total
        @test maximum(vol) == 2.0f0
        println("  ✓ Boundary insert: $labeled voxels (expected $expected_total)")
    end
    
    @testset "3. Heat-GDT full pipeline: diffusion → mask → label" begin
        # Create a 128³ volume with uniform diffusivity (ensures heat spreads)
        D_vol = ones(Float32, 128, 128, 128)
        
        cx, cy, cz = 64, 64, 64
        # Use more diffusion steps and a very low theta to ensure some voxels pass threshold
        mask = run_heatgdt_cpu(D_vol, cx, cy, cz; K=80, theta=0.0001f0, tau=0.00001f0)
        
        @test size(mask) == (64, 64, 64)
        voxels = count(mask .> 0)
        @test voxels > 0  # Should segment something with uniform diffusivity
        println("  ✓ Heat-GDT produced $voxels voxels")
        
        # Now insert into label volume
        label_vol = zeros(Float32, 128, 128, 128)
        insert_patch!(label_vol, mask, cx, cy, cz; label_val=1.0f0)
        
        label_count = count(label_vol .> 0)
        @test label_count == voxels  # All patch voxels should map to full volume
        
        # Verify labels are within expected spatial range
        labeled_indices = findall(label_vol .> 0)
        if !isempty(labeled_indices)
            x_coords = [I[1] for I in labeled_indices]
            y_coords = [I[2] for I in labeled_indices]
            z_coords = [I[3] for I in labeled_indices]
            
            @test minimum(x_coords) >= 32  # cx - 32
            @test maximum(x_coords) <= 96  # cx + 32 - 1
            @test minimum(y_coords) >= 32
            @test maximum(y_coords) <= 96
            @test minimum(z_coords) >= 32
            @test maximum(z_coords) <= 96
            
            println("  ✓ Labels correctly placed in range [$(minimum(x_coords)):$(maximum(x_coords)), $(minimum(y_coords)):$(maximum(y_coords)), $(minimum(z_coords)):$(maximum(z_coords))]")
        else
            error("No labeled voxels found — heat diffusion may have failed")
        end
    end
    
    @testset "4. Diffusivity barrier traps heat inside" begin
        # Strong barrier should prevent heat from escaping
        D_vol = ones(Float32, 128, 128, 128)
        # Very strong barrier: near-zero diffusivity wall
        for x in 1:128, y in 1:128, z in 1:128
            r = sqrt((x-64)^2 + (y-64)^2 + (z-64)^2)
            if 10 < r < 12
                D_vol[x, y, z] = 0.0001f0
            end
        end
        
        mask = run_heatgdt_cpu(D_vol, 64, 64, 64; K=40, theta=0.001f0)
        voxels = count(mask .> 0)
        
        # Check: all segmented voxels should be within the barrier (radius < 12)
        patch_center = 33  # Center of 64³ patch
        for ci in findall(mask .> 0)
            r_patch = sqrt((ci[1]-patch_center)^2 + (ci[2]-patch_center)^2 + (ci[3]-patch_center)^2)
            @test r_patch < 14  # Allow some tolerance
        end
        
        println("  ✓ Barrier traps heat: $voxels voxels all within barrier")
    end
    
    @testset "5. Multiple lesion IDs don't overwrite each other" begin
        vol = zeros(Float32, 128, 128, 128)
        
        # Insert lesion 1 at (40, 64, 64)
        mask1 = zeros(UInt8, 64, 64, 64)
        mask1[30:36, 30:36, 30:36] .= 1
        insert_patch!(vol, mask1, 40, 64, 64; label_val=1.0f0)
        
        # Insert lesion 2 at (90, 64, 64)
        mask2 = zeros(UInt8, 64, 64, 64)
        mask2[30:36, 30:36, 30:36] .= 1
        insert_patch!(vol, mask2, 90, 64, 64; label_val=2.0f0)
        
        count_1 = count(vol .== 1.0f0)
        count_2 = count(vol .== 2.0f0)
        
        @test count_1 == 7^3
        @test count_2 == 7^3
        @test count_1 + count_2 == count(vol .> 0)  # No overlap
        println("  ✓ Two lesions: $count_1 + $count_2 voxels, no overlap")
    end
end

println("\n" * "=" ^ 70)
println("  All Label Save Roundtrip Tests Complete!")
println("=" ^ 70)
