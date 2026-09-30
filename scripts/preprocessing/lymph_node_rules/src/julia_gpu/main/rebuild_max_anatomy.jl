using NIfTI
using HDF5
include("MaxAnatomyBuilder.jl")
using .MaxAnatomyBuilder

case_dir = "/mnt/big/project_ssd/project_ssd/lymph_node_rules/data/processed_cases_restored/FDM_DPI-2024-7-KRN_Prostata_bimodal__PETPSMA_0__Pat44"
h5_path = joinpath(case_dir, "primary_masks.h5")
ct_path = joinpath(case_dir, "Fixed_CT_Volume.nii.gz")
results_h5 = "/mnt/big/project_ssd/project_ssd/lymph_node_rules/data/validation_batch_v118/Pat44_results_V118.h5"

println("Loading base masks from H5...")
base_masks = Dict{String, AbstractArray}()
h5open(h5_path, "r") do f
    if haskey(f, "masks")
        for name in keys(f["masks"])
            base_masks[name] = read(f["masks"][name])
        end
    end
end

println("Loading gen masks from results H5...")
gen_masks = Dict{String, AbstractArray}()
h5open(results_h5, "r") do f
    for name in keys(f)
        if typeof(f[name]) <: HDF5.Dataset
            gen_masks[name] = read(f[name])
        end
    end
end

println("Loading CT header...")
ref_hdr = deepcopy(niread(ct_path).header)

println("Rebuilding Max Anatomy...")
MaxAnatomyBuilder.build_and_save_max_anatomy(base_masks, gen_masks, ref_hdr, case_dir)

println("Copying to mrb_temp...")
cp(joinpath(case_dir, "max_anatomy.nii.gz"), "/mnt/big/project_ssd/project_ssd/lymph_node_rules/data/mrb_temp/max_anatomy.nii.gz", force=true)
cp(joinpath(case_dir, "max_anatomy_labels.json"), "/mnt/big/project_ssd/project_ssd/lymph_node_rules/data/mrb_temp/max_anatomy_labels.json", force=true)

println("Done!")
