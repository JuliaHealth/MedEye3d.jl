using HDF5, JSON, KernelAbstractions, NPZ, CUDA
include("main/DagVm.jl")
using .DagVm

println("======================================================")
println("Running Full Julia GPU Pipeline on Pat44")
println("MegaKernel / DAG VM Architecture")
println("======================================================")

h5_path = "data/processed_cases_restored/FDM_DPI-2024-7-KRN_Prostata_bimodal__PETPSMA_0__Pat44/primary_masks.h5"

if !isfile(h5_path)
    println("ERROR: HDF5 file not found at ", h5_path)
    exit(1)
end

backend = CUDA.functional() ? CUDABackend() : CPU()
println("Backend selected: ", typeof(backend))

out_dir = "/tmp/julia_pre_overlap_pat44"
rm(out_dir, force=true, recursive=true)
mkpath(out_dir)

# Run the DAG pipeline bypassing overlap resolution (evaluates all areas directly)
result = DagVm.run_pipeline(h5_path; resolve_overlaps=true, backend=backend)

n_saved = 0
for (k, v) in result
    arr = collect(v)
    if eltype(arr) == Bool
        arr = UInt8.(arr)
    elseif eltype(arr) != UInt8
        arr = UInt8.(arr .> 0)
    end
    # Permute dims to match Python (ZYX)
    NPZ.npzwrite("$(out_dir)/$(k).npy", permutedims(arr, (3, 2, 1)))
    global n_saved += 1
    arr = nothing
    GC.gc(true)
end

println("======================================================")
println("Successfully executed DAG VM and saved $(n_saved) masks to $(out_dir)")
println("======================================================")
