using Pkg
Pkg.activate(".")

using MedEye3d
using MedEye3d.MedImages
using MedEye3d.LesionAssociation
using MedEye3d.BoneSubsegmentation
using HDF5
using KernelAbstractions
using CUDA

data_dir_pat6 = joinpath(pwd(), "data", "pat_6_files")
mask_path = joinpath(data_dir_pat6, "PET_Lesions_0.seg.nrrd")
ts_nrrd_path = joinpath(data_dir_pat6, "TS_all_Segmentation_0.seg.nrrd")
output_h5 = joinpath(data_dir_pat6, "Bone_Subsegments_0.h5")

println("Loading mask...")
mask_vol, _, _, _, mask_spacing, _, _, _ = MedEye3d.MedImages.load_image(mask_path)
first_mask = Float32.(mask_vol)

println("Loading TS atlas...")
ts_atlas, ts_sizes = LesionAssociation.load_nrrd_labelmap(ts_nrrd_path)
ts_atlas_aligned = reverse(ts_atlas, dims=2)
bone_atlas = Float32.(ts_atlas_aligned .> 0)

println("Extracting lesions...")
lids = filter(x -> x > 0, unique(first_mask))

println("Precomputing Bone Subsegments...")
h5open(output_h5, "w") do file
    for lid in lids
        println("Processing Lesion ID: $lid")
        surf, marr = BoneSubsegmentation.generate_bone_subsegments(first_mask, bone_atlas, mask_spacing, Int(lid))
        
        # Convert bit arrays to UInt8 to save in HDF5
        file["lesion_$(Int(lid))_surf"] = UInt8.(surf)
        file["lesion_$(Int(lid))_marr"] = UInt8.(marr)
    end
end

println("Precomputation complete. Saved to $output_h5")
