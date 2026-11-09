#!/usr/bin/env julia
"""
    generate_modality_diffusivities.jl

Computes modality-specific edge detection and Heat-GDT diffusivity:
- CT-only diffusivity (`diffusivity_ct`): PET channel masked to 0.0
- PET-only diffusivity (`diffusivity_pet`): CT channel masked to -1024 HU (air baseline)

Usage:
    CUDA_VISIBLE_DEVICES=1 julia --project=/tmp/backup_repo_clean/tmp_juliahelpnet scripts/preprocessing/generate_modality_diffusivities.jl <h5_path_or_dir> [--ct-only | --pet-only | --both]
"""

using Pkg

const HEATGDT_DIR = get(ENV, "HEATGDT_DIR", "/tmp/backup_repo_clean/tmp_juliahelpnet/HeatGDT")
const HEATGDT_PROJECT = dirname(HEATGDT_DIR)

if isdir(HEATGDT_PROJECT) && isfile(joinpath(HEATGDT_PROJECT, "Project.toml"))
    Pkg.activate(HEATGDT_PROJECT)
end

using Lux
using LuxCUDA
using CUDA
using ComponentArrays
using JLD2
using NNlib
using Statistics
using HDF5

if isfile(joinpath(HEATGDT_DIR, "architectures", "hed3d_ultra.jl"))
    include(joinpath(HEATGDT_DIR, "architectures", "hed3d_ultra.jl"))
else
    error("Architecture file hed3d_ultra.jl not found at $HEATGDT_DIR/architectures/hed3d_ultra.jl")
end

const gdev = gpu_device()
const cdev = cpu_device()

const CT_CLIP = (-1024.0f0, 1500.0f0)
const CT_MEAN = -200.0f0
const CT_STD  = 400.0f0
const PET_CLIP_MAX = 20.0f0

const CHAMPION_GAMMA = 10.095886f0
const CHAMPION_EPS_D = 1.0f-5

function compute_diffusivity(edge_map; epsilon_D::Float32=CHAMPION_EPS_D, gamma::Float32=CHAMPION_GAMMA)
    D = epsilon_D .+ (1.0f0 - epsilon_D) .* exp.(-gamma .* edge_map)
    return clamp.(D, epsilon_D, 1.0f0)
end

function sliding_window_edge_inference(ct::Array{Float32,3}, pet::Array{Float32,3}, ts::Array{Float32,3},
                                        model, ps, st; patch_size::Int=64, stride::Int=32)
    sz = size(ct)
    edge_volume = zeros(Float32, sz)
    weight_volume = zeros(Float32, sz)
    
    input_buf = Array{Float32}(undef, patch_size, patch_size, patch_size, 3, 1)
    patch_count = 0
    total_patches = length(1:stride:sz[3]) * length(1:stride:sz[2]) * length(1:stride:sz[1])
    t_start = time()
    
    for z in 1:stride:sz[3]
        for y in 1:stride:sz[2]
            for x in 1:stride:sz[1]
                x1, y1, z1 = x, y, z
                x2 = min(x1 + patch_size - 1, sz[1])
                y2 = min(y1 + patch_size - 1, sz[2])
                z2 = min(z1 + patch_size - 1, sz[3])
                
                if x2 - x1 + 1 < patch_size; x1 = max(1, x2 - patch_size + 1); end
                if y2 - y1 + 1 < patch_size; y1 = max(1, y2 - patch_size + 1); end
                if z2 - z1 + 1 < patch_size; z1 = max(1, z2 - patch_size + 1); end
                
                ct_patch = @view ct[x1:x2, y1:y2, z1:z2]
                pet_patch = @view pet[x1:x2, y1:y2, z1:z2]
                ts_patch = @view ts[x1:x2, y1:y2, z1:z2]
                
                @views input_buf[:,:,:,1,1] .= (clamp.(ct_patch, CT_CLIP[1], CT_CLIP[2]) .- CT_MEAN) ./ CT_STD
                @views input_buf[:,:,:,2,1] .= clamp.(pet_patch, 0.0f0, PET_CLIP_MAX) ./ PET_CLIP_MAX
                @views input_buf[:,:,:,3,1] .= ts_patch ./ 117.0f0
                
                x_gpu = input_buf |> gdev
                edge_out, _ = model(x_gpu, ps, st)
                edge_cpu = Array(edge_out)
                
                edge_volume[x1:x2, y1:y2, z1:z2] .+= @view edge_cpu[:,:,:,1,1]
                weight_volume[x1:x2, y1:y2, z1:z2] .+= 1.0f0
                
                patch_count += 1
                if patch_count % 100 == 0 || patch_count == total_patches
                    elapsed = time() - t_start
                    rate = patch_count / elapsed
                    eta = (total_patches - patch_count) / rate
                    println("    Patch $patch_count/$total_patches ($(round(rate, digits=1))/s, ETA $(round(eta, digits=0))s)")
                end
                
                x_gpu = nothing
                edge_out = nothing
                if patch_count % 32 == 0
                    GC.gc(false)
                    CUDA.reclaim()
                end
            end
        end
    end
    
    edge_volume ./= max.(weight_volume, 1.0f0)
    return edge_volume
end

function get_tp_groups(h5_file)
    all_keys = collect(keys(h5_file))
    meta_keys = Set(["_meta_", "ATLAS", "CENTROIDS", "BONE_SUBSEG"])
    tp_groups = filter(k -> !(k in meta_keys), all_keys)
    
    function tp_sort_key(name)
        if name == "BASELINE"; return 0; end
        m = match(r"_(\d+)\.", name)
        return m !== nothing ? parse(Int, m.captures[1]) : 999
    end
    sort!(tp_groups, by=tp_sort_key)
    return tp_groups
end

function process_file(h5_path::String; do_ct=true, do_pet=true)
    println("═══════════════════════════════════════════════════════════════")
    println("  Generating Modality-Specific Diffusivity Fields")
    println("═══════════════════════════════════════════════════════════════")
    println("  HDF5: $h5_path")
    println("  CT Only: $do_ct | PET Only: $do_pet")
    println("  GPU Device: $(CUDA.device())")
    
    jld_path = joinpath(HEATGDT_DIR, "champion_pretrained_hed3d_ultra.jld2")
    if !isfile(jld_path)
        error("Model checkpoint not found: $jld_path")
    end
    
    println("\n[1/3] Loading HED3D-Ultra edge network...")
    model = HED3D_Ultra(3)
    ckpt = JLD2.load(jld_path)
    ps_full = ComponentArray(ckpt["ps"]) |> gdev
    ps = haskey(ps_full, :edge_net) ? ps_full.edge_net : ps_full
    st_full = Lux.testmode(ckpt["st"]) |> gdev
    st = haskey(st_full, :edge_net) ? st_full.edge_net : st_full
    println("  ✓ Model loaded on $(CUDA.device())")
    
    h5_file = h5open(h5_path, "r+")
    is_preflipped = haskey(h5_file, "_meta_/preflipped") && read(h5_file["_meta_/preflipped"]) == 1
    tp_groups = get_tp_groups(h5_file)
    
    has_atlas = haskey(h5_file, "ATLAS/max_anatomy")
    atlas_vol = has_atlas ? Float32.(read(h5_file["ATLAS/max_anatomy"])) : nothing
    
    for (tp_idx, group_name) in enumerate(tp_groups)
        println("\n── Group: $group_name ──")
        group_keys = collect(keys(h5_file[group_name]))
        
        ct_key = nothing
        pet_key = nothing
        ts_key = nothing
        for k in group_keys
            if startswith(k, "Fixed_CT_Volume") || (occursin("CT", k) && !occursin("PET", k) && !occursin("Lesion", k) && !occursin("anatomy", k) && !occursin("prostate", k))
                ct_key = k
            end
            if startswith(k, "SUV_PET") || (occursin("PET", k) && occursin("SUV", k)) || occursin("SPECT", k) && !occursin("CT", k) && !occursin("Lesion", k)
                pet_key = k
            end
            if k == "max_anatomy.nii.gz"
                ts_key = k
            end
        end
        
        if ct_key === nothing || pet_key === nothing
            println("  ⚠️ Missing CT ($ct_key) or PET ($pet_key), skipping.")
            continue
        end
        
        ct_vol = Float32.(read(h5_file["$group_name/$ct_key"]))
        pet_vol = Float32.(read(h5_file["$group_name/$pet_key"]))
        ts_vol = if ts_key !== nothing
            Float32.(read(h5_file["$group_name/$ts_key"]))
        elseif atlas_vol !== nothing
            copy(atlas_vol)
        else
            zeros(Float32, size(ct_vol))
        end
        
        if is_preflipped
            ct_native = reverse(ct_vol, dims=2)
            pet_native = reverse(pet_vol, dims=2)
            ts_native = reverse(ts_vol, dims=2)
        else
            ct_native = ct_vol
            pet_native = pet_vol
            ts_native = ts_vol
        end
        
        # 1. CT Only: mask PET channel to 0.0
        if do_ct
            println("  --> Computing CT-only edges (masking PET channel to 0.0)...")
            pet_masked = zeros(Float32, size(pet_native))
            t0 = time()
            edge_ct = sliding_window_edge_inference(ct_native, pet_masked, ts_native, model, ps, st)
            diff_ct = compute_diffusivity(edge_ct)
            diff_ct_save = is_preflipped ? reverse(diff_ct, dims=2) : diff_ct
            
            key_ct = "$group_name/diffusivity_ct"
            if haskey(h5_file, key_ct)
                delete_object(h5_file, key_ct)
            end
            h5_file[key_ct] = diff_ct_save
            flush(h5_file)
            println("  ✓ Saved $key_ct in $(round(time() - t0, digits=1))s: mean=$(round(mean(diff_ct), digits=4))")
            edge_ct = nothing; diff_ct = nothing; diff_ct_save = nothing
        end
        
        # 2. PET Only: mask CT channel to air baseline (-1024 HU)
        if do_pet
            println("  --> Computing PET-only edges (masking CT channel to -1024 HU)...")
            ct_masked = fill(-1024.0f0, size(ct_native))
            t0 = time()
            edge_pet = sliding_window_edge_inference(ct_masked, pet_native, ts_native, model, ps, st)
            diff_pet = compute_diffusivity(edge_pet)
            diff_pet_save = is_preflipped ? reverse(diff_pet, dims=2) : diff_pet
            
            key_pet = "$group_name/diffusivity_pet"
            if haskey(h5_file, key_pet)
                delete_object(h5_file, key_pet)
            end
            h5_file[key_pet] = diff_pet_save
            flush(h5_file)
            println("  ✓ Saved $key_pet in $(round(time() - t0, digits=1))s: mean=$(round(mean(diff_pet), digits=4))")
            edge_pet = nothing; diff_pet = nothing; diff_pet_save = nothing
        end
        
        ct_vol = nothing; pet_vol = nothing; ts_vol = nothing
        ct_native = nothing; pet_native = nothing; ts_native = nothing
        GC.gc(true)
        CUDA.reclaim()
    end
    
    close(h5_file)
    println("\n✅ Done processing $h5_path")
end

if abspath(PROGRAM_FILE) == @__FILE__
    target = isempty(ARGS) ? "/workspaces/MedEye3d.jl/data/pat_6_files/preprocessed_volumes.h5" : ARGS[1]
    h5_path = isdir(target) ? joinpath(target, "preprocessed_volumes.h5") : target
    
    do_ct = !any(a -> a == "--pet-only", ARGS)
    do_pet = !any(a -> a == "--ct-only", ARGS)
    
    process_file(h5_path; do_ct=do_ct, do_pet=do_pet)
end
