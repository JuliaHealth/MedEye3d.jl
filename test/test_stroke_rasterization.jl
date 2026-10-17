using Test
using MedEye3d
using MedEye3d.StrokeRasterization
using CUDA

println("================================================================================")
println("         STROKE RASTERIZATION (CONTINUOUS THICK LINE) TEST SUITE")
println("================================================================================")
println("CUDA Functional: ", CUDA.functional())
if CUDA.functional()
    println("CUDA Device: ", CUDA.name(CUDA.device()))
end

@testset "StrokeRasterization (KernelAbstractions)" begin

    @testset "Single Point Stamp (Circle)" begin
        mask = zeros(UInt8, 100, 100)
        center = (50, 50)
        radius = 5
        rasterize_polyline!(mask, [center], radius, UInt8(1), use_gpu=false)
        
        # Verify center is filled
        @test mask[50, 50] == 1
        # Verify points at distance <= radius are filled
        @test mask[50 + radius, 50] == 1
        @test mask[50, 50 + radius] == 1
        @test mask[50 - radius, 50] == 1
        @test mask[50, 50 - radius] == 1
        # Verify points outside radius are 0
        @test mask[50 + radius + 1, 50] == 0
        @test mask[50, 50 + radius + 1] == 0
        
        # Exact circle formula check
        for y in 1:100, x in 1:100
            expected = ((x - 50)^2 + (y - 50)^2 <= radius^2) ? 1 : 0
            @test mask[x, y] == expected
        end
        println("✓ Single point stamp test passed")
    end

    @testset "Continuous Line Segment (No Gaps)" begin
        mask = zeros(UInt8, 200, 200)
        p1 = (20, 20)
        p2 = (180, 100)
        radius = 4
        rasterize_thick_line!(mask, p1, p2, radius, UInt8(2), use_gpu=false)
        
        @test mask[20, 20] == 2
        @test mask[180, 100] == 2
        
        # Check that every intermediate point along the line is covered (no gaps)
        for t in 0.0:0.01:1.0
            ix = round(Int, 20 + t * (180 - 20))
            iy = round(Int, 20 + t * (100 - 20))
            @test mask[ix, iy] == 2
        end
        println("✓ Continuous line segment test passed")
    end

    @testset "Fast Mouse Drag Interpolation" begin
        # Simulate fast mouse movement with large jump between recorded points
        mask = zeros(UInt8, 512, 512)
        # Fast moves jumping 100 pixels per frame
        sampled_points = [(50, 50), (150, 100), (250, 300), (450, 450)]
        radius = 5
        rasterize_polyline!(mask, sampled_points, radius, UInt8(1), use_gpu=false)
        
        # Check continuity between sampled_points[1] and sampled_points[2]
        for t in 0.0:0.02:1.0
            ix = round(Int, 50 + t * 100)
            iy = round(Int, 50 + t * 50)
            @test mask[ix, iy] == 1
        end
        
        # Check continuity between sampled_points[2] and sampled_points[3]
        for t in 0.0:0.02:1.0
            ix = round(Int, 150 + t * 100)
            iy = round(Int, 100 + t * 200)
            @test mask[ix, iy] == 1
        end
        
        println("✓ Fast mouse drag interpolation test passed (zero gaps)")
    end

    @testset "Erase Mode (Value = 0)" begin
        # Start with a fully solid block of lesion label 1
        mask = ones(UInt8, 100, 100)
        # Erase a diagonal thick stroke across it
        p1 = (10, 10)
        p2 = (90, 90)
        radius = 6
        rasterize_thick_line!(mask, p1, p2, radius, UInt8(0), use_gpu=false)
        
        # Verify the stroke path has been erased to 0
        @test mask[10, 10] == 0
        @test mask[50, 50] == 0
        @test mask[90, 90] == 0
        
        # Verify untouched areas remain 1
        @test mask[10, 90] == 1
        @test mask[90, 10] == 1
        println("✓ Erase mode test passed")
    end

    @testset "Boundary Clipping" begin
        mask = zeros(UInt8, 100, 100)
        # Stroke crossing boundary outside [1, 100]
        p1 = (1, 1)
        p2 = (100, 1)
        radius = 10
        # Must not error or throw OutOfBoundsError
        rasterize_thick_line!(mask, p1, p2, radius, UInt8(1), use_gpu=false)
        @test mask[1, 1] == 1
        @test mask[50, 1] == 1
        @test mask[100, 1] == 1
        println("✓ Boundary clipping test passed")
    end

    if CUDA.functional()
        @testset "GPU vs CPU Exact Match" begin
            p1 = (30, 40)
            p2 = (480, 450)
            radius = 8
            
            mask_cpu = zeros(UInt8, 512, 512)
            rasterize_thick_line!(mask_cpu, p1, p2, radius, UInt8(3), use_gpu=false)
            
            mask_gpu = CUDA.zeros(UInt8, 512, 512)
            rasterize_thick_line!(mask_gpu, p1, p2, radius, UInt8(3), use_gpu=true)
            mask_gpu_cpu = Array(mask_gpu)
            
            @test mask_cpu == mask_gpu_cpu
            @test count(mask_cpu .> 0) == count(mask_gpu_cpu .> 0)
            println("✓ GPU vs CPU exact match test passed (Dice = 1.0, 0 mismatched pixels)")
        end
    end

    @testset "Performance Benchmark (512x512 Slice)" begin
        mask = zeros(UInt8, 512, 512)
        p1 = (50, 50)
        p2 = (450, 400)
        radius = 5
        
        # Warmup
        rasterize_thick_line!(mask, p1, p2, radius, UInt8(1), use_gpu=false)
        
        # Benchmark function
        function benchmark_fn(f, runs=50)
            f() # warmup
            times = Float64[]
            for _ in 1:runs
                push!(times, @elapsed f())
            end
            return minimum(times)
        end
        
        t_cpu = benchmark_fn(() -> rasterize_thick_line!(mask, p1, p2, radius, UInt8(1), use_gpu=false))
        println("\n--- Benchmark (512x512 Slice, 400px Diagonal Stroke) ---")
        println("  KernelAbstractions CPU: ", round(t_cpu * 1000, digits=4), " ms")
        @test t_cpu < 0.005 # Less than 5 ms (interactive is < 16.6ms)
        
        if CUDA.functional()
            mask_gpu = CUDA.zeros(UInt8, 512, 512)
            t_gpu = benchmark_fn(() -> rasterize_thick_line!(mask_gpu, p1, p2, radius, UInt8(1), use_gpu=true))
            println("  KernelAbstractions GPU: ", round(t_gpu * 1000, digits=4), " ms")
            @test t_gpu < 0.005
        end
    end

    @testset "Brush Circle Display (Draw & Erase, 50% Opacity)" begin
        MockCalcDims = @NamedTuple{imageTextureWidth::Int, imageTextureHeight::Int}
        MockPanelState = @NamedTuple{
            currentDisplayedSlice::Int,
            calcDimsStruct::MockCalcDims,
            spacingsValue::Vector{NTuple{3, Float64}}
        }
        sp = [(2.0, 2.0, 3.0)]
        mock_states = [
            (currentDisplayedSlice = 100, calcDimsStruct = (imageTextureWidth = 512, imageTextureHeight = 512), spacingsValue = sp),
            (currentDisplayedSlice = 100, calcDimsStruct = (imageTextureWidth = 512, imageTextureHeight = 512), spacingsValue = sp),
            (currentDisplayedSlice = 250, calcDimsStruct = (imageTextureWidth = 512, imageTextureHeight = 326), spacingsValue = sp),
            (currentDisplayedSlice = 260, calcDimsStruct = (imageTextureWidth = 512, imageTextureHeight = 326), spacingsValue = sp),
        ]

        # 1. Draw mode on Axial panel 1 (radius = 5 voxels)
        verts_draw = MedEye3d.Measurements.compute_brush_circle_vertices(
            mock_states, 1, (250.0f0, 260.0f0, 100.0f0), 5; is_erase = false
        )
        # 64 segments * 6 vertices/segment * 6 floats/vertex = 2304 floats
        @test length(verts_draw) == 64 * 6 * 6
        # Verify 50% opacity (alpha == 0.5f0) and green color (0.3, 0.85, 0.3) on all vertices
        for v_idx in 1:6:length(verts_draw)
            @test verts_draw[v_idx + 2] ≈ 0.3f0
            @test verts_draw[v_idx + 3] ≈ 0.85f0
            @test verts_draw[v_idx + 4] ≈ 0.3f0
            @test verts_draw[v_idx + 5] == 0.5f0
        end

        # 2. Erase mode on Axial panel 1 (radius = 8 voxels)
        verts_erase = MedEye3d.Measurements.compute_brush_circle_vertices(
            mock_states, 1, (250.0f0, 260.0f0, 100.0f0), 8; is_erase = true
        )
        @test length(verts_erase) == 64 * 6 * 6
        # Verify 50% opacity (alpha == 0.5f0) and red color (1.0, 0.3, 0.3)
        for v_idx in 1:6:length(verts_erase)
            @test verts_erase[v_idx + 2] ≈ 1.0f0
            @test verts_erase[v_idx + 3] ≈ 0.3f0
            @test verts_erase[v_idx + 4] ≈ 0.3f0
            @test verts_erase[v_idx + 5] == 0.5f0
        end

        # Verify radius scales when stroke_width increases from 5 to 8
        u_vals_5 = [verts_draw[i] for i in 1:6:length(verts_draw)]
        u_vals_8 = [verts_erase[i] for i in 1:6:length(verts_erase)]
        span_5 = maximum(u_vals_5) - minimum(u_vals_5)
        span_8 = maximum(u_vals_8) - minimum(u_vals_8)
        @test span_8 > span_5

        # 3. Sagittal (panel 3) and Coronal (panel 4) views
        verts_sag = MedEye3d.Measurements.compute_brush_circle_vertices(
            mock_states, 3, (250.0f0, 260.0f0, 100.0f0), 6; is_erase = false
        )
        @test length(verts_sag) == 64 * 6 * 6
        @test verts_sag[6] == 0.5f0

        verts_cor = MedEye3d.Measurements.compute_brush_circle_vertices(
            mock_states, 4, (250.0f0, 260.0f0, 100.0f0), 6; is_erase = true
        )
        @test length(verts_cor) == 64 * 6 * 6
        @test verts_cor[6] == 0.5f0
        println("✓ Brush circle display test passed (Draw & Erase at 50% opacity)")
    end

end
