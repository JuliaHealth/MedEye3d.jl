using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

# GPU isolation (GPU 0 for interactive viewer)
if !haskey(ENV, "CUDA_VISIBLE_DEVICES")
    ENV["CUDA_VISIBLE_DEVICES"] = "0"
end

# HDF5 thread-safe configuration (must be set before first HDF5 open)
ENV["HDF5_USE_FILE_LOCKING"] = "FALSE"

# Configure Vulkan ICD if NVIDIA driver is present
if isfile("/etc/vulkan/icd.d/nvidia_icd.json") && !haskey(ENV, "VK_ICD_FILENAMES")
    ENV["VK_ICD_FILENAMES"] = "/etc/vulkan/icd.d/nvidia_icd.json"
    ENV["VK_DRIVER_FILES"] = "/etc/vulkan/icd.d/nvidia_icd.json"
end

using MedEye3d
using MedEye3d.AppMain

# Resolve target path
repo_root = normpath(joinpath(@__DIR__, "..", ".."))
raw_arg = length(ARGS) >= 1 ? ARGS[1] : ""

function resolve_h5_path(arg::String, root::String)
    isempty(arg) && return ""
    isfile(arg) && return abspath(arg)
    dir_h5 = joinpath(arg, "preprocessed_volumes.h5")
    isfile(dir_h5) && return abspath(dir_h5)
    
    # Check numeric patient ID (e.g., 23 -> Pat23)
    pat_str = occursin(r"^\d+$", arg) ? "Pat$arg" : arg
    bimodal_h5 = joinpath(root, "data", "cases", "Prostata_bimodal", pat_str, "preprocessed_volumes.h5")
    isfile(bimodal_h5) && return abspath(bimodal_h5)
    
    data_h5 = joinpath(root, "data", pat_str, "preprocessed_volumes.h5")
    isfile(data_h5) && return abspath(data_h5)
    
    return ""
end

preprocessed_h5 = resolve_h5_path(raw_arg, repo_root)

if isempty(preprocessed_h5)
    # Default fallback: Pat23, then pat_6_files
    pat23_h5 = joinpath(repo_root, "data", "cases", "Prostata_bimodal", "Pat23", "preprocessed_volumes.h5")
    pat6_h5 = joinpath(repo_root, "data", "pat_6_files", "preprocessed_volumes.h5")
    if isfile(pat23_h5)
        preprocessed_h5 = pat23_h5
    elseif isfile(pat6_h5)
        preprocessed_h5 = pat6_h5
    else
        error("preprocessed_volumes.h5 not found. Specify a patient ID (e.g., 23, Pat23) or a valid file path.")
    end
end

println("==================================================")
println("  Starting MedEye3d Interactive Session")
println("  Dataset: $preprocessed_h5")
println("  GPU:     CUDA_VISIBLE_DEVICES=$(get(ENV, "CUDA_VISIBLE_DEVICES", "not set"))")
println("==================================================")

AppMain.launch_from_h5(preprocessed_h5; quad=true)
