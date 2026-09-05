using Test
using NIfTI
using Statistics

@testset "PET SUV Calibration" begin
    cases_dir = joinpath(@__DIR__, "..", "data", "cases", "psma_patient_all_tp")
    for i in 0:2
        pet_path = joinpath(cases_dir, "SUV_PET_Image_$i.nii.gz")
        @test isfile(pet_path)
        nii = niread(pet_path)
        raw = Float32.(nii.raw)
        pos = raw[raw .> 0]
        @test !isempty(pos)
        
        mx = maximum(pos)
        # Verify calibrated SUV ranges (max should be tens to ~170 SUV, NOT 350,000 Bq/mL)
        @test mx > 10.0f0
        @test mx < 250.0f0
        
        # Verify typical 95th percentile is modest (< 15 SUV)
        p95 = quantile(pos, 0.95)
        @test p95 < 15.0
    end
end

@testset "MakieEventHandlers Windowing Config" begin
    include(joinpath(@__DIR__, "..", "scripts", "lib", "SceneHierarchy.jl"))
    
    # Check that SceneHierarchy correctly infers modalities for 7 timepoints
    case_dir = joinpath(@__DIR__, "..", "data", "cases", "psma_patient_all_tp")
    studies = SceneHierarchy.parse_studies_from_hierarchy(case_dir)
    @test length(studies) == 7
    
    expected_modalities = ["CT", "CT", "CT", "T2", "ADC", "DWI", "T1"]
    for (idx, exp_mod) in enumerate(expected_modalities)
        @test studies[idx][1] == exp_mod
    end
end
