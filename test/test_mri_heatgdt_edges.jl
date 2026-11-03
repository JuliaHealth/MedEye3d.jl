using Test
using HDF5
using MedEye3d

println("==================================================")
println("  Test MRI Diffusivity & Heat-GDT Quad Integration")
println("==================================================")

h5_path = "/workspaces/MedEye3d.jl/data/cases/Prostata_bimodal/Pat23/preprocessed_volumes.h5"
@test isfile(h5_path)

# 1. Verify HDF5 contains genuine edge-based diffusivity for all MRI groups
HDF5.h5open(h5_path, "r") do h5
    base_diff = Float32.(read(h5["BASELINE/diffusivity"]))
    @test size(base_diff) == (512, 512, 284)
    println("BASELINE/diffusivity: min=$(minimum(base_diff)), max=$(maximum(base_diff)), mean=$(sum(base_diff)/length(base_diff))")
    
    # Check TP 1..4 (the quad MRI modalities)
    for tp in 1:4
        grp = "TFM_Transform_FollowUp_to_Baseline_$tp.tfm"
        @test haskey(h5, grp)
        @test haskey(h5[grp], "diffusivity")
        
        mri_diff = Float32.(read(h5["$grp/diffusivity"]))
        @test size(mri_diff) == (512, 512, 284)
        
        # Verify it is NOT identical to BASELINE (CT) diffusivity!
        max_diff = maximum(abs.(mri_diff .- base_diff))
        @test max_diff > 0.1f0
        
        # Verify edge contrast exists: min near eps_D, max == 1.0, mean < 1.0
        d_min = minimum(mri_diff)
        d_max = maximum(mri_diff)
        d_mean = sum(mri_diff) / length(mri_diff)
        println("$grp: min=$d_min, max=$d_max, mean=$d_mean, max_diff_from_ct=$max_diff")
        
        @test d_min < 0.001f0  # Strong edge barrier
        @test d_max ≈ 1.0f0    # Free flow in flat areas
        @test d_mean < 0.99f0  # Edges are present across volume
    end
end

# 2. Test MakieEventHandlers diffusivity reloading and memory caching
MEH = MedEye3d.SegmentationDisplay.MakieEventHandlers
IC = MedEye3d.InferenceClient

MEH.h5_save_path_ref[] = h5_path
empty!(MEH._tp_diffusivity_cache)
IC.clear_heatgdt_diffusivity!()

@test IC._heatgdt_diffusivity_tp[] == -1
@test IC.is_heatgdt_available() == false

# Reload TP 1 (T2)
MEH.reload_heatgdt_diffusivity_for_tp!(1)
@test IC._heatgdt_diffusivity_tp[] == 1
@test IC.is_heatgdt_available() == true
@test haskey(MEH._tp_diffusivity_cache, 1)
diff_t2 = IC._heatgdt_diffusivity[]
@test diff_t2 !== nothing
@test size(diff_t2) == (512, 512, 284)

# Reload TP 2 (ADC) — should switch to TP 2 and cache it
MEH.reload_heatgdt_diffusivity_for_tp!(2)
@test IC._heatgdt_diffusivity_tp[] == 2
@test haskey(MEH._tp_diffusivity_cache, 2)
diff_adc = IC._heatgdt_diffusivity[]
@test maximum(abs.(diff_t2 .- diff_adc)) > 0.01f0  # T2 and ADC diffusivity are distinct!

# Switching back to TP 1 should use cache directly
t_switch = @elapsed MEH.reload_heatgdt_diffusivity_for_tp!(1)
@test IC._heatgdt_diffusivity_tp[] == 1
println("Diffusivity cache hit switch time: $(round(t_switch * 1000, digits=2)) ms")
@test t_switch < 0.05  # Blazing fast from cache

println("\n✅ All MRI Diffusivity & Heat-GDT Quad Integration tests passed!")
