using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

# HDF5 thread-safe configuration (must be set before first HDF5 open)
ENV["HDF5_USE_FILE_LOCKING"] = "FALSE"

# Configure Vulkan ICD if NVIDIA driver is present
if isfile("/etc/vulkan/icd.d/nvidia_icd.json") && !haskey(ENV, "VK_ICD_FILENAMES")
    ENV["VK_ICD_FILENAMES"] = "/etc/vulkan/icd.d/nvidia_icd.json"
    ENV["VK_DRIVER_FILES"] = "/etc/vulkan/icd.d/nvidia_icd.json"
end

using MedEye3d
using MedEye3d.AppMain

data_dir_pat6 = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "..", "..", "data", "pat_6_files")
preprocessed_h5 = isfile(data_dir_pat6) ? data_dir_pat6 : joinpath(data_dir_pat6, "preprocessed_volumes.h5")

if !isfile(preprocessed_h5)
    error("preprocessed_volumes.h5 not found in $data_dir_pat6. Run: julia scripts/preprocessing/preprocess_dataset.jl $data_dir_pat6")
end

println("==================================================")
println("  Starting MedEye3d Interactive Session")
println("  Dataset: $preprocessed_h5")
println("==================================================")

AppMain.launch_from_h5(preprocessed_h5; quad=true)
