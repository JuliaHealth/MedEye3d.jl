using Pkg
Pkg.activate("/workspaces/MedEye3d.jl")

using MedImages
using HDF5
using Base.Filesystem

function process_directory(data_dir, out_dir)
    mkpath(out_dir)
    
    files = readdir(data_dir)
    nrrd_files = filter(f -> endswith(f, ".nrrd"), files)
    
    for f in nrrd_files
        in_path = joinpath(data_dir, f)
        out_path = joinpath(out_dir, replace(f, ".nrrd" => ".hdf5"))
        out_path = replace(out_path, ".seg.hdf5" => ".seg.hdf5")
        
        println("Converting $f to HDF5...")
        try
            img = MedImages.Load_and_save.load_image(in_path, "")
            
            f_h5 = h5open(out_path, "w")
            uid = MedImages.save_med_image(f_h5, "images", img)
            close(f_h5)
            println("  Saved to $out_path with UID $uid")
        catch e
            println("  Failed to process $f: $e")
        end
    end

    # Copy JSON files to the output directory as well
    json_files = filter(f -> endswith(f, ".json") || endswith(f, ".txt") || endswith(f, ".csv"), files)
    for f in json_files
        cp(joinpath(data_dir, f), joinpath(out_dir, f), force=true)
        println("Copied $f")
    end
    
    println("Conversion completed.")
end

if length(ARGS) < 2
    println("Usage: julia convert_to_hdf5.jl <input_dir> <output_dir>")
else
    process_directory(ARGS[1], ARGS[2])
end
