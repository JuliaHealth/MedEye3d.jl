using Test
using MedEye3d.LesionMetadataWindow
using MedEye3d.LesionMetadataWindow: compute_lesion_suv_metrics, precompute_all_suv_metrics!,
                                      get_lesion_suv_metrics, _get_suv_max, _get_suv_mean, _get_suv_peak,
                                      compute_lesion_suv_string, parse_suv_fields, _lesion_suv_metrics_cache

@testset "Per-Lesion SUV Metrics Calculation" begin
    # Create synthetic 3D volumes (20 x 20 x 20)
    pet_vol = fill(1.0f0, 20, 20, 20)
    mask_vol = zeros(Int16, 20, 20, 20)

    # Simulate whole-body high-uptake background (e.g. bladder, brain, injection site)
    pet_vol[1:3, 1:3, 1:3] .= 75.0f0  # Bladder with SUV 75.0 (outside any lesion)
    pet_vol[18:20, 18:20, 18:20] .= 45.0f0 # Salivary gland with SUV 45.0

    # Define Lesion 1 in mask (voxels 8:12, 8:12, 8:12 -> 5x5x5 = 125 voxels)
    mask_vol[8:12, 8:12, 8:12] .= 1
    # Assign specific PET values to Lesion 1: base = 4.0, with a gradient up to 14.5
    for (i, idx) in enumerate(findall(mask_vol .== 1))
        pet_vol[idx] = 4.0f0 + Float32(i % 10) * 1.05f0
    end
    # Force an exact maximum at a specific voxel
    pet_vol[10, 10, 10] = 15.5f0
    
    # Calculate expected max and mean strictly for Lesion 1
    l1_vox = findall(mask_vol .== 1)
    l1_vals = [pet_vol[idx] for idx in l1_vox]
    expected_l1_max = maximum(l1_vals) # 15.5
    expected_l1_mean = sum(l1_vals) / length(l1_vals)

    # Define Lesion 2 in mask (voxels 14:16, 14:16, 14:16 -> 3x3x3 = 27 voxels)
    mask_vol[14:16, 14:16, 14:16] .= 2
    for (i, idx) in enumerate(findall(mask_vol .== 2))
        pet_vol[idx] = 2.0f0 + Float32(i) * 0.2f0
    end
    l2_vox = findall(mask_vol .== 2)
    l2_vals = [pet_vol[idx] for idx in l2_vox]
    expected_l2_max = maximum(l2_vals)
    expected_l2_mean = sum(l2_vals) / length(l2_vals)

    @testset "Direct per-lesion SUV metrics" begin
        # 1. Lesion 1 metrics must NOT see whole body uptake (SUV 75.0)
        m1 = compute_lesion_suv_metrics(pet_vol, mask_vol, 1; sp_x=1.0, sp_y=1.0, sp_z=1.0)
        @test m1.max ≈ expected_l1_max atol=1e-3
        @test m1.max ≈ 15.5f0
        @test m1.max < 70.0f0 # Critical: Whole-body bladder uptake must NEVER pollute lesion SUVmax
        @test m1.mean ≈ expected_l1_mean atol=1e-3
        @test m1.peak <= m1.max
        @test m1.peak > 0.0f0

        # 2. Lesion 2 metrics
        m2 = compute_lesion_suv_metrics(pet_vol, mask_vol, 2; sp_x=1.0, sp_y=1.0, sp_z=1.0)
        @test m2.max ≈ expected_l2_max atol=1e-3
        @test m2.mean ≈ expected_l2_mean atol=1e-3
        @test m2.max < m1.max

        # 3. Non-existent lesion returns 0.0
        m_empty = compute_lesion_suv_metrics(pet_vol, mask_vol, 999)
        @test m_empty.max == 0.0f0
        @test m_empty.mean == 0.0f0
        @test m_empty.peak == 0.0f0

        # 4. Invalid lesion ID <= 0 returns 0.0
        m_inv = compute_lesion_suv_metrics(pet_vol, mask_vol, 0)
        @test m_inv.max == 0.0f0
    end

    @testset "Precomputing all SUV metrics across volume" begin
        test_tp = 99
        empty!(_lesion_suv_metrics_cache)
        precompute_all_suv_metrics!(pet_vol, mask_vol, test_tp)

        @test haskey(_lesion_suv_metrics_cache, (test_tp, 1))
        @test haskey(_lesion_suv_metrics_cache, (test_tp, 2))
        @test !haskey(_lesion_suv_metrics_cache, (test_tp, 999))

        cached_l1 = _lesion_suv_metrics_cache[(test_tp, 1)]
        @test cached_l1.max ≈ expected_l1_max atol=1e-3
        @test cached_l1.mean ≈ expected_l1_mean atol=1e-3

        cached_l2 = _lesion_suv_metrics_cache[(test_tp, 2)]
        @test cached_l2.max ≈ expected_l2_max atol=1e-3
        @test cached_l2.mean ≈ expected_l2_mean atol=1e-3

        # Test getter functions utilizing cache
        @test _get_suv_max(1, test_tp) ≈ expected_l1_max atol=1e-3
        @test _get_suv_mean(1, test_tp) ≈ expected_l1_mean atol=1e-3
        @test _get_suv_peak(1, test_tp) ≈ cached_l1.peak atol=1e-3
    end

    @testset "SUV String Parsing & Formatting" begin
        # Formatted string with Max, Mean, Peak, and reference organs
        s = "Max: 15.5 ; Mean: 8.7 ; Peak: 14.2 ; Blood: 1.8 ; Liver: 2.5 ; Parotid: 3.2"
        f = parse_suv_fields(s)
        @test f["max"] ≈ 15.5f0
        @test f["mean"] ≈ 8.7f0
        @test f["peak"] ≈ 14.2f0
        @test f["blood"] ≈ 1.8f0
        @test f["liver"] ≈ 2.5f0
        @test f["parotid"] ≈ 3.2f0

        # Plain numeric string
        f_num = parse_suv_fields("12.4")
        @test f_num["max"] ≈ 12.4f0

        # Empty string
        @test isempty(parse_suv_fields(""))
    end

    @testset "Schema includes both SUV max and SUV mean" begin
        schema = LesionMetadataWindow.load_schema()
        schema_dict = Dict(q.short => q for q in schema)
        @test haskey(schema_dict, "SUV max")
        @test haskey(schema_dict, "SUV mean")
        @test schema_dict["SUV max"].default_answer == "0.0"
        @test schema_dict["SUV mean"].default_answer == "0.0"
        @test "Quantitative" in schema_dict["SUV max"].categories
        @test "Quantitative" in schema_dict["SUV mean"].categories
    end
end
