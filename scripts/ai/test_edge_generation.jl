using Pkg
Pkg.activate("/mnt/big/project_ssd/project_ssd/semiautomatic/JuliaHELPNet")

using NIfTI
using Statistics

# Include the actual script which exports/defines generate_edge_image
include("/mnt/big/project_ssd/project_ssd/MedEye3d.jl/scripts/ai/generate_edge_image.jl")

function run_test()
    data_dir = "/mnt/big/project_ssd/project_ssd/MedEye3d.jl/data/pat_6_files/"
    
    ct_path = joinpath(data_dir, "Fixed_CT_Volume_0.nii.gz")
    pet_path = joinpath(data_dir, "SUV_PET_Image_0.nii.gz")
    ts_path = joinpath(data_dir, "TS_all_Segmentation_0.nii.gz")
    
    output_prefix = joinpath(data_dir, "test_edge_out_0")
    
    println("Running edge generation test on timepoint 0...")
    generate_edge_image(ct_path, pet_path, ts_path, output_prefix)
    
    println("\nVerifying outputs...")
    edge_path = output_prefix * "_edge.nii.gz"
    diff_path = output_prefix * "_diffusivity.nii.gz"
    
    ct_nii = niread(ct_path)
    edge_nii = niread(edge_path)
    diff_nii = niread(diff_path)
    
    sz_ct = size(ct_nii.raw)
    sz_edge = size(edge_nii.raw)
    sz_diff = size(diff_nii.raw)
    
    println("Original CT shape: ", sz_ct)
    println("Edge volume shape: ", sz_edge)
    println("Diffusivity shape: ", sz_diff)
    
    if sz_ct == sz_edge == sz_diff
        println("SUCCESS: Output shapes match input shapes!")
    else
        println("ERROR: Shape mismatch!")
    end
    
    edge_data = Float32.(edge_nii.raw)
    diff_data = Float32.(diff_nii.raw)
    
    println("\nStatistics:")
    println("Edge Volume       - Min: ", minimum(edge_data), ", Max: ", maximum(edge_data), ", Mean: ", mean(edge_data))
    println("Diffusivity Vol   - Min: ", minimum(diff_data), ", Max: ", maximum(diff_data), ", Mean: ", mean(diff_data))
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_test()
end
