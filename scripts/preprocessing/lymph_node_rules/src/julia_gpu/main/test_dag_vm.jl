using HDF5, JSON, KernelAbstractions, CUDA, NPZ
include("DagVm.jl"); using .DagVm

h5 = "data/processed_cases_restored/FDM_DPI-2024-7-KRN_Prostata_bimodal__PETPSMA_0__Pat44/primary_masks.h5"
result = DagVm.run_pipeline(h5; resolve_overlaps=true)

h5_out = "final_results_gpu.h5"
println("\nSaving $(length(result)) raw masks to $h5_out...")
h5open(h5_out, "w") do f
    for (k, v) in result
        try
            arr = collect(v)
            if eltype(arr) == Bool; arr = UInt8.(arr)
            elseif eltype(arr) != UInt8; arr = UInt8.(arr .> 0); end
            if ndims(arr) == 3
                f[k] = arr
            end
        catch e
            println("ERROR saving $k: $e")
        end
    end
end
println("Done saving HDF5.")

mrb_out = "data/Pat44_Clean_MRB.mrb"
println("\nExporting clean 3D Slicer MRB to $mrb_out...")
export_script = joinpath(@__DIR__, "export_clean_mrb.py")
run(`python3 $export_script --h5 $h5 --results $h5_out --output-mrb $mrb_out`)
println("Pipeline and MRB generation complete.")

# Overwrite the old combined file so the user sees the refresh
combined_mrb = "data/Pat44_Julia_Combined.mrb"
println("Copying to $combined_mrb to ensure user sees the refreshed file...")
cp(mrb_out, combined_mrb, force=true)

# Overwrite the validation_live_fresh MRB
validation_mrb = "data/validation_live_fresh/FDM_DPI-2024-7-KRN_Prostata_bimodal__PETPSMA_0__Pat44/FDM_DPI-2024-7-KRN_Prostata_bimodal__PETPSMA_0__Pat44_Combined.mrb"
println("Copying to $validation_mrb to ensure user sees the refreshed file...")
cp(mrb_out, validation_mrb, force=true)
cp("data/Pat44_Clean_MRB.mrb", "data/patient_mrbs/axillary_validation_44.mrb", force=true)
cp("data/Pat44_Clean_MRB.mrb", "data/patient_mrbs/axillary_validation_44.mrb", force=true)
