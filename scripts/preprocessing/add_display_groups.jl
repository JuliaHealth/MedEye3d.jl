#!/usr/bin/env julia
# Add _DISPLAY groups to existing preprocessed_volumes.h5
# Reads native-res volumes, resamples to 2× in-plane, writes _DISPLAY groups

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using HDF5
using MedImages

const HIRES_FACTOR = 2.0
const H5_PATH = joinpath(@__DIR__, "..", "..", "data", "pat_6_files", "preprocessed_volumes.h5")

function main()
    println("Opening $H5_PATH in read-write mode...")
    h5 = h5open(H5_PATH, "r+")
    
    native_groups = [k for k in keys(h5) if !endswith(k, "_DISPLAY")]
    println("Found $(length(native_groups)) native groups: $native_groups")
    
    for group_name in native_groups
        display_group = group_name * "_DISPLAY"
        
        # Skip if already exists
        if haskey(h5, display_group)
            println("  $display_group already exists, skipping")
            continue
        end
        
        println("\nProcessing group: $group_name → $display_group")
        datasets = keys(h5[group_name])
        
        for ds_name in datasets
            t_start = time_ns()
            println("  Reading $group_name/$ds_name...")
            
            ds = h5["$group_name/$ds_name"]
            vol = Float32.(read(ds))
            
            # Read spacing from attributes
            sp = if haskey(HDF5.attributes(ds), "spacing")
                Tuple(Float64.(read(HDF5.attributes(ds)["spacing"])))
            else
                (0.9765625, 0.9765625, 3.0)  # fallback
            end
            
            # Read origin and direction if available
            origin = if haskey(HDF5.attributes(ds), "origin")
                Tuple(Float64.(read(HDF5.attributes(ds)["origin"])))
            else
                (0.0, 0.0, 0.0)
            end
            
            direction = if haskey(HDF5.attributes(ds), "direction")
                Tuple(Float64.(read(HDF5.attributes(ds)["direction"])))
            else
                (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
            end
            
            # Determine interpolation: nearest for masks, linear for CT/PET
            is_mask = occursin("Lesions", ds_name) || occursin("Mask", ds_name) || occursin("seg", lowercase(ds_name))
            interp = is_mask ? MedImages.Nearest_neighbour_en : MedImages.Linear_en
            
            # Compute display spacing (2× in-plane)
            display_sp = (sp[1] / HIRES_FACTOR, sp[2] / HIRES_FACTOR, sp[3])
            
            println("    Native: $(size(vol)) @ $sp → Display: $display_sp ($(is_mask ? "NN" : "Linear"))")
            
            # Create MedImage and resample
            img = MedImage(
                voxel_data=vol,
                spacing=sp,
                origin=origin,
                direction=direction,
                image_type=MedImages.MedImage_data_struct.MRI_type,
                image_subtype=MedImages.MedImage_data_struct.CT_subtype,
                patient_id="hires"
            )
            
            resampled = MedImages.resample_to_spacing(img, display_sp, interp)
            display_vol = Float32.(resampled.voxel_data)
            
            # Write to _DISPLAY group
            MedImages.save_med_image(h5, display_group, ds_name, resampled)
            
            t_ms = (time_ns() - t_start) / 1e6
            println("    Saved $display_group/$ds_name: $(size(display_vol)) in $(round(t_ms/1000, digits=1))s")
            flush(stdout)
        end
    end
    
    close(h5)
    println("\nDone! Verifying...")
    
    # Verify
    h5v = h5open(H5_PATH, "r")
    display_groups = [k for k in keys(h5v) if endswith(k, "_DISPLAY")]
    println("Display groups: $(length(display_groups))")
    for g in display_groups
        for dk in keys(h5v[g])
            println("  $g/$dk: $(size(h5v[g][dk]))")
        end
    end
    close(h5v)
    
    # Show file size
    fsize = filesize(H5_PATH)
    println("\nFinal file size: $(round(fsize / 1e9, digits=2)) GB")
end

main()
