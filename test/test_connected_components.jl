using Test
using MedEye3d
using MedEye3d.ConnectedComponents
using CUDA

println("================================================================================")
println("           CONNECTED COMPONENTS (LCC) TEST & BENCHMARK SUITE")
println("================================================================================")
println("CUDA Functional: ", CUDA.functional())
if CUDA.functional()
    println("CUDA Device: ", CUDA.name(CUDA.device()))
end

@testset "ConnectedComponents (KernelAbstractions & SimpleITK Baseline)" begin

    # Helper function to call SimpleITK baseline in Python
    function sitk_largest_connected_component(mask::Array{UInt8, 3}; connectivity::Int=26)
        # Write array to temp binary / NIfTI or evaluate via python script
        tmp_in = tempname() * "_in.raw"
        tmp_out = tempname() * "_out.raw"
        
        # Save raw binary bytes with Fortran (Julia) order
        write(tmp_in, mask)
        
        dims_str = "$(size(mask, 1)),$(size(mask, 2)),$(size(mask, 3))"
        fully_conn = connectivity == 26 ? "True" : "False"
        
        py_cmd = """
import numpy as np
import SimpleITK as sitk

dims = tuple(map(int, '$dims_str'.split(',')))
raw = np.fromfile('$tmp_in', dtype=np.uint8)
data_xyz = raw.reshape(dims, order='F')

if np.count_nonzero(data_xyz) == 0:
    out_xyz = np.zeros_like(data_xyz)
else:
    data_zyx = np.transpose(data_xyz, (2, 1, 0))
    img = sitk.GetImageFromArray(data_zyx)
    cc = sitk.ConnectedComponent(img, $fully_conn)
    relabeled = sitk.RelabelComponent(cc)
    out_zyx = (sitk.GetArrayFromImage(relabeled) == 1).astype(np.uint8)
    out_xyz = np.transpose(out_zyx, (2, 1, 0))

flat_out = out_xyz.flatten(order='F')
flat_out.tofile('$tmp_out')
"""
        run(`python3 -c $py_cmd`)
        
        res_bytes = read(tmp_out)
        res = reshape(reinterpret(UInt8, res_bytes), size(mask))
        
        rm(tmp_in, force=true)
        rm(tmp_out, force=true)
        return res
    end

    function dice_score(a::AbstractArray, b::AbstractArray)
        a_bool = a .> 0
        b_bool = b .> 0
        intersection = count(a_bool .& b_bool)
        total = count(a_bool) + count(b_bool)
        if total == 0
            return 1.0
        end
        return (2.0 * intersection) / total
    end

    @testset "Edge Case 1: Empty Mask" begin
        empty_vol = zeros(UInt8, 64, 64, 64)
        
        res_cpu = extract_largest_connected_component(empty_vol, use_gpu=false)
        @test count(res_cpu .> 0) == 0
        
        if CUDA.functional()
            res_gpu = extract_largest_connected_component(empty_vol, use_gpu=true)
            @test count(res_gpu .> 0) == 0
        end
        
        res_sitk = sitk_largest_connected_component(empty_vol)
        @test count(res_sitk .> 0) == 0
        println("✓ Empty mask test passed")
    end

    @testset "Edge Case 2: Single Connected Component" begin
        single_vol = zeros(UInt8, 64, 64, 64)
        single_vol[20:30, 20:30, 20:30] .= 1
        
        init_count = count(single_vol .> 0)
        res_cpu = extract_largest_connected_component(single_vol, use_gpu=false)
        @test count(res_cpu .> 0) == init_count
        @test res_cpu == single_vol
        
        if CUDA.functional()
            res_gpu = extract_largest_connected_component(single_vol, use_gpu=true)
            @test res_gpu == single_vol
        end
        
        res_sitk = sitk_largest_connected_component(single_vol)
        @test res_sitk == single_vol
        @test dice_score(res_cpu, res_sitk) == 1.0
        println("✓ Single component test passed (Dice = 1.0)")
    end

    @testset "Multi-Component: Disconnected Spheres" begin
        multi_vol = zeros(UInt8, 64, 64, 64)
        # Small sphere (radius 3) at (15, 15, 15) -> ~123 voxels
        for z in 1:64, y in 1:64, x in 1:64
            if (x-15)^2 + (y-15)^2 + (z-15)^2 <= 9
                multi_vol[x, y, z] = 1
            end
            # Medium sphere (radius 5) at (25, 45, 25) -> ~515 voxels
            if (x-25)^2 + (y-45)^2 + (z-25)^2 <= 25
                multi_vol[x, y, z] = 1
            end
            # Large sphere (radius 8) at (45, 45, 45) -> ~2109 voxels (LARGEST)
            if (x-45)^2 + (y-45)^2 + (z-45)^2 <= 64
                multi_vol[x, y, z] = 1
            end
        end
        
        total_initial = count(multi_vol .> 0)
        println("Initial multi-component voxels: ", total_initial)
        
        res_cpu = extract_largest_connected_component(multi_vol, use_gpu=false)
        res_sitk = sitk_largest_connected_component(multi_vol)
        
        @test count(res_cpu .> 0) < total_initial
        @test count(res_cpu .> 0) == count(res_sitk .> 0)
        @test dice_score(res_cpu, res_sitk) == 1.0
        @test res_cpu == res_sitk
        
        if CUDA.functional()
            res_gpu = extract_largest_connected_component(multi_vol, use_gpu=true)
            @test res_gpu == res_sitk
            @test dice_score(res_gpu, res_sitk) == 1.0
        end
        
        println("✓ Multi-component spheres test passed! (KA GPU/CPU vs SimpleITK: Dice = 1.000, $(count(res_cpu .> 0)) voxels)")
    end

    @testset "Complex Topologies & Noise Specks" begin
        complex_vol = zeros(UInt8, 64, 64, 64)
        # Main irregular lesion
        for z in 20:44, y in 20:44, x in 20:44
            if (x-32)^2/12^2 + (y-32)^2/8^2 + (z-32)^2/10^2 <= 1.0
                complex_vol[x, y, z] = 1
            end
        end
        
        # Add 50 isolated random 1-voxel and 2-voxel noise specks
        import Random
        Random.seed!(42)
        for _ in 1:50
            rx, ry, rz = rand(1:64), rand(1:64), rand(1:64)
            if (rx-32)^2 + (ry-32)^2 + (rz-32)^2 > 250
                complex_vol[rx, ry, rz] = 1
            end
        end
        
        res_cpu = extract_largest_connected_component(complex_vol, use_gpu=false)
        res_sitk = sitk_largest_connected_component(complex_vol)
        
        @test res_cpu == res_sitk
        @test dice_score(res_cpu, res_sitk) == 1.0
        
        if CUDA.functional()
            res_gpu = extract_largest_connected_component(complex_vol, use_gpu=true)
            @test res_gpu == res_sitk
            @test dice_score(res_gpu, res_sitk) == 1.0
        end
        
        println("✓ Complex lesion with random noise specks passed! (KA vs SimpleITK: Dice = 1.000)")
    end

    @testset "Performance Benchmark: KA GPU vs KA CPU vs SimpleITK" begin
        bench_vol = zeros(UInt8, 64, 64, 64)
        for z in 1:64, y in 1:64, x in 1:64
            if (x-32)^2 + (y-32)^2 + (z-32)^2 <= 20^2
                bench_vol[x, y, z] = 1
            end
            if (x-10)^2 + (y-10)^2 + (z-10)^2 <= 5^2
                bench_vol[x, y, z] = 1
            end
        end
        
        println("\n--- Benchmarking 64x64x64 Patch ---")
        
        # Benchmark function with warmup and multiple runs
        function benchmark_fn(f, runs=10)
            f() # warmup
            times = Float64[]
            for _ in 1:runs
                push!(times, @elapsed f())
            end
            return minimum(times)
        end
        
        t_cpu = benchmark_fn(() -> extract_largest_connected_component(bench_vol, use_gpu=false))
        println("  KernelAbstractions CPU:   ", round(t_cpu * 1000, digits=2), " ms")
        
        if CUDA.functional()
            t_gpu = benchmark_fn(() -> extract_largest_connected_component(bench_vol, use_gpu=true))
            println("  KernelAbstractions GPU:   ", round(t_gpu * 1000, digits=2), " ms")
        end
        
        t_sitk = benchmark_fn(() -> sitk_largest_connected_component(bench_vol), 5)
        println("  SimpleITK Baseline (C++): ", round(t_sitk * 1000, digits=2), " ms")
        
        @test true
    end
end
