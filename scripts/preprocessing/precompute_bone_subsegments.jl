using Pkg
Pkg.activate(".")

using MedEye3d
using MedImages
using MedEye3d.LesionAssociation
using MedEye3d.BoneSubsegmentation
using HDF5
using KernelAbstractions

data_dir_pat6 = joinpath(@__DIR__, "..", "..", "data", "pat_6_files")
mask_path = joinpath(data_dir_pat6, "PET_Lesions_0.seg.nrrd")
ts_nrrd_path = joinpath(data_dir_pat6, "TS_all_Segmentation_0.seg.nrrd")
output_h5 = joinpath(data_dir_pat6, "Bone_Subsegments_0.h5")

println("Loading baseline CT and mask...")
baseline_ct = MedImages.load_image(joinpath(data_dir_pat6, "Fixed_CT_Volume_0.nii.gz"), "CT")
seg_raw = MedImages.load_image(joinpath(data_dir_pat6, "PET_Lesions_0.nii.gz"), "CT")
seg_res = MedImages.resample_to_image(baseline_ct, seg_raw, MedImages.Nearest_neighbour_en)
first_mask = reverse(Float32.(seg_res.voxel_data), dims=2)
mask_spacing = Tuple(Float64.(baseline_ct.spacing))

println("Loading TS atlas...")
ts_atlas, ts_sizes = LesionAssociation.load_nrrd_labelmap(ts_nrrd_path)
ts_atlas_aligned = reverse(ts_atlas, dims=2)
ts_names = LesionAssociation.parse_nrrd_segment_names(ts_nrrd_path)

bone_keywords = ["femur", "hip", "vertebra", "rib", "sacrum", "clavicula", "humerus", "scapula", "sternum", "skull", "palate", "bone", "spine"]
bone_labels = Set{Int}()
for (k, v) in ts_names
    v_low = lowercase(v)
    if any(kw -> occursin(kw, v_low), bone_keywords)
        push!(bone_labels, k)
    end
end
println("Identified bone labels in TS: $(collect(bone_labels))")

# True Bone Atlas = TS bone labels OR CT bone threshold (> 180 HU)
ct_vox = Float32.(baseline_ct.voxel_data)
ct_aligned = reverse(ct_vox, dims=2)
bone_atlas = Float32.(in.(ts_atlas_aligned, Ref(bone_labels)) .| (ct_aligned .>= 180.0f0))

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
