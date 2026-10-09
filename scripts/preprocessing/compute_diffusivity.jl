"""
    compute_diffusivity.jl — Computes NN-based edge detection + diffusivity for all TPs in an HDF5 file.
    
    Usage:
        julia compute_diffusivity.jl <data_dir>
    
    Reads CT, PET, and max_anatomy (TotalSeg) from the preprocessed_volumes.h5 file,
    runs the HED3D-Wide edge network (sliding-window inference), computes the diffusivity
    field D(x) = ε + (1-ε)·exp(-γ·E(x)), and stores it back in the same HDF5 file
    as a `diffusivity` dataset in each TP group.
    
    The HED3D-Wide model checkpoint and architecture are loaded from:
        /mnt/big/project_ssd/project_ssd/semiautomatic/JuliaHELPNet/HeatGDT/
"""

using Pkg
Pkg.activate("/mnt/big/project_ssd/project_ssd/semiautomatic/JuliaHELPNet")

using Lux
using LuxCUDA
using CUDA
using ComponentArrays
using JLD2
using NNlib
using Statistics
using CUDA
using HDF5
using JSON

include("/mnt/big/project_ssd/project_ssd/semiautomatic/JuliaHELPNet/HeatGDT/architectures/hed3d_wide.jl")

# Device helpers
const gdev = gpu_device()
const cdev = cpu_device()

"""
    compute_diffusivity(edge_map; epsilon_D=0.01f0, gamma=5.0f0)

Convert edge probability map E ∈ [0,1] to diffusivity field D:
    D(x) = ε_D + (1 - ε_D) · exp(-γ · E(x))

High edges → low diffusivity (heat blocked), low edges → high diffusivity (heat flows freely).
"""
function compute_diffusivity(edge_map; epsilon_D::Float32=0.01f0, gamma::Float32=5.0f0)
    return epsilon_D .+ (1.0f0 - epsilon_D) .* exp.(-gamma .* edge_map)
end

"""
    sliding_window_edge_inference(ct, pet, ts, model, ps, st; patch_size=64, stride=32)

Run HED3D-Wide edge network on a full volume using sliding-window inference.
Inputs must be in NATIVE (un-flipped) space, Float32.

Returns Float32 edge volume (same size as inputs), values ∈ [0,1].
"""
function sliding_window_edge_inference(ct::Array{Float32,3}, pet::Array{Float32,3}, ts::Array{Float32,3},
                                        model, ps, st; patch_size::Int=64, stride::Int=32)
    sz = size(ct)
    @assert sz == size(pet) "CT and PET shapes must match! Got CT=$(sz), PET=$(size(pet))"
    @assert sz == size(ts) "CT and TotalSeg shapes must match! Got CT=$(sz), TS=$(size(ts))"
    
    # Handle PET/CT resolution mismatch: resample PET to CT grid if needed
    if size(pet) != sz
        println("  PET resolution $(size(pet)) differs from CT $(sz), resampling PET to CT grid...")
        pet_5d = reshape(pet, size(pet)..., 1, 1)
        target_size = (sz[1], sz[2], sz[3])
        pet_resampled = NNlib.upsample_trilinear(pet_5d; size=target_size)
        pet = dropdims(pet_resampled, dims=(4, 5))
        println("  PET resampled to $(size(pet))")
    end
    
    edge_volume = zeros(Float32, sz)
    weight_volume = zeros(Float32, sz)
    
    # Pre-allocate CPU input buffer
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
                
                # Clamp to ensure full patch size
                if x2 - x1 + 1 < patch_size; x1 = max(1, x2 - patch_size + 1); end
                if y2 - y1 + 1 < patch_size; y1 = max(1, y2 - patch_size + 1); end
                if z2 - z1 + 1 < patch_size; z1 = max(1, z2 - patch_size + 1); end
                
                ct_patch = @view ct[x1:x2, y1:y2, z1:z2]
                pet_patch = @view pet[x1:x2, y1:y2, z1:z2]
                ts_patch = @view ts[x1:x2, y1:y2, z1:z2]
                
                # z-score normalization per patch (matching training)
                ct_m = mean(ct_patch)
                ct_s = std(ct_patch) + 1f-6
                
                # Fill pre-allocated buffer
                @views input_buf[:,:,:,1,1] .= (ct_patch .- ct_m) ./ ct_s
                @views input_buf[:,:,:,2,1] .= pet_patch
                @views input_buf[:,:,:,3,1] .= ts_patch ./ 117.0f0
                
                x_gpu = input_buf |> gdev
                
                # Forward pass in test mode
                edge_out, _ = model(x_gpu, ps, st)
                
                edge_cpu = Array(edge_out)
                edge_patch_out = @view edge_cpu[:,:,:,1,1]
                
                edge_volume[x1:x2, y1:y2, z1:z2] .+= edge_patch_out
                weight_volume[x1:x2, y1:y2, z1:z2] .+= 1.0f0
                
                patch_count += 1
                if patch_count % 50 == 0 || patch_count == total_patches
                    elapsed = time() - t_start
                    rate = patch_count / elapsed
                    eta = (total_patches - patch_count) / rate
                    println("  Patch $patch_count/$total_patches ($(round(rate, digits=1))/s, ETA $(round(eta, digits=0))s)")
                end
                
                # Cleanup GPU memory explicitly
                x_gpu = nothing
                edge_out = nothing
                if patch_count % 16 == 0
                    GC.gc(false)
                    CUDA.reclaim()
                end
            end
        end
    end
    
    # Average overlapping regions
    edge_volume ./= max.(weight_volume, 1.0f0)
    return edge_volume
end

"""
    get_tp_groups(h5_file) → Vector{String}

Returns the TP group names in order: BASELINE first, then TFM_* groups sorted by index.
"""
function get_tp_groups(h5_file)
    all_keys = collect(keys(h5_file))
    meta_keys = Set(["_meta_", "ATLAS", "CENTROIDS", "BONE_SUBSEG"])
    tp_groups = filter(k -> !(k in meta_keys), all_keys)
    
    # Sort: BASELINE first, then TFM_* by numeric suffix
    function tp_sort_key(name)
        if name == "BASELINE"
            return 0
        end
        # Extract numeric suffix from TFM_Transform_FollowUp_to_Baseline_N.tfm
        m = match(r"_(\d+)\.", name)
        if m !== nothing
            return parse(Int, m.captures[1])
        end
        return 999
    end
    
    sort!(tp_groups, by=tp_sort_key)
    return tp_groups
end

function main()
    if isempty(ARGS)
        error("Usage: julia compute_diffusivity.jl <data_dir>")
    end
    data_dir = abspath(ARGS[1])
    h5_path = joinpath(data_dir, "preprocessed_volumes.h5")
    
    if !isfile(h5_path)
        error("HDF5 file not found: $h5_path")
    end
    
    println("═══════════════════════════════════════════════════")
    println("  Computing HED3D-Wide Edge Detection + Diffusivity    ")
    println("═══════════════════════════════════════════════════")
    println("  HDF5: $h5_path")
    
    # ── Load model ──────────────────────────────────────────
    println("\n[1/3] Loading HED3D-Wide edge network (best pretrained model)...")
    model = HED3D_Wide(3)
    
    jld_path = "/mnt/big/project_ssd/project_ssd/semiautomatic/JuliaHELPNet/HeatGDT/champion_pretrained_hed3d_wide.jld2"
    if !isfile(jld_path)
        error("Model checkpoint not found: $jld_path")
    end
    
    ckpt = JLD2.load(jld_path)
    ps_full = ComponentArray(ckpt["ps"]) |> gdev
    # Unwrap nested checkpoint format (training wrapper wraps under .edge_net)
    ps = haskey(ps_full, :edge_net) ? ps_full.edge_net : ps_full
    st_full = Lux.testmode(ckpt["st"]) |> gdev
    st = haskey(st_full, :edge_net) ? st_full.edge_net : st_full
    
    println("  Model loaded: $(length(ps)) parameters on $(gdev)")
    
    # ── Process each TP ─────────────────────────────────────
    println("\n[2/3] Processing timepoints...")
    
    h5_file = h5open(h5_path, "r+")
    
    # Check preflipped status
    is_preflipped = haskey(h5_file, "_meta_/preflipped") && read(h5_file["_meta_/preflipped"]) == 1
    println("  Data is preflipped: $is_preflipped")
    
    tp_groups = get_tp_groups(h5_file)
    println("  Found $(length(tp_groups)) timepoint groups: $(tp_groups)")
    
    # Check if max_anatomy atlas is available (used as TotalSeg channel)
    has_atlas = haskey(h5_file, "ATLAS/max_anatomy")
    atlas_vol = has_atlas ? Float32.(read(h5_file["ATLAS/max_anatomy"])) : nothing
    if has_atlas
        println("  Using global ATLAS/max_anatomy as TotalSeg channel ($(size(atlas_vol)))")
    end
    
    for (tp_idx, group_name) in enumerate(tp_groups)
        tp_i = tp_idx - 1
        println("\n  ── TP $tp_i ($group_name) ──")
        
        # Check if diffusivity already exists
        diff_key = "$group_name/diffusivity"
        if haskey(h5_file, diff_key)
            println("    ⏭  Diffusivity already exists, skipping")
            continue
        end
        
        # Find CT volume in this group
        group_keys = collect(keys(h5_file[group_name]))
        ct_key = nothing
        pet_key = nothing
        ts_key = nothing  # per-TP max_anatomy
        
        for k in group_keys
            if startswith(k, "Fixed_CT_Volume") || (occursin("CT", k) && !occursin("PET", k) && !occursin("Lesion", k) && !occursin("anatomy", k) && !occursin("prostate", k))
                ct_key = k
            end
            if startswith(k, "SUV_PET") || (occursin("PET", k) && occursin("SUV", k))
                pet_key = k
            end
            if k == "max_anatomy.nii.gz"
                ts_key = k
            end
        end
        
        if ct_key === nothing
            println("    ⚠️  No CT volume found in $group_name, skipping")
            continue
        end
        if pet_key === nothing
            println("    ⚠️  No PET volume found in $group_name, skipping")
            continue
        end
        
        println("    CT:  $ct_key")
        println("    PET: $pet_key")
        
        # Load volumes (already preflipped in HDF5)
        ct_vol = Float32.(read(h5_file["$group_name/$ct_key"]))
        pet_vol = Float32.(read(h5_file["$group_name/$pet_key"]))
        
        # Use per-TP max_anatomy if available, otherwise global atlas
        ts_vol = if ts_key !== nothing
            println("    TS:  $ts_key (per-TP)")
            Float32.(read(h5_file["$group_name/$ts_key"]))
        elseif atlas_vol !== nothing
            println("    TS:  ATLAS/max_anatomy (global)")
            copy(atlas_vol)
        else
            println("    TS:  NONE — using zeros as TotalSeg channel")
            zeros(Float32, size(ct_vol))
        end
        
        # Un-flip for NN inference (model was trained on non-flipped data)
        if is_preflipped
            ct_native = reverse(ct_vol, dims=2)
            pet_native = reverse(pet_vol, dims=2)
            ts_native = reverse(ts_vol, dims=2)
        else
            ct_native = ct_vol
            pet_native = pet_vol
            ts_native = ts_vol
        end
        
        println("    Volume size: $(size(ct_native))")
        println("    Running sliding-window edge inference...")
        
        t_edge = time()
        edge_map = sliding_window_edge_inference(ct_native, pet_native, ts_native, model, ps, st)
        elapsed_edge = time() - t_edge
        println("    Edge inference done in $(round(elapsed_edge, digits=1))s")
        println("    Edge stats: min=$(round(minimum(edge_map), digits=4)) max=$(round(maximum(edge_map), digits=4)) mean=$(round(mean(edge_map), digits=4))")
        
        # Compute diffusivity
        diff_vol = compute_diffusivity(edge_map)
        println("    Diffusivity stats: min=$(round(minimum(diff_vol), digits=4)) max=$(round(maximum(diff_vol), digits=4)) mean=$(round(mean(diff_vol), digits=4))")
        
        # Re-flip to match HDF5 convention (preflipped)
        if is_preflipped
            diff_vol_flipped = reverse(diff_vol, dims=2)
        else
            diff_vol_flipped = diff_vol
        end
        
        # Store in HDF5 without compression to avoid inflate errors
        println("    Saving to $diff_key (uncompressed)...")
        h5_file[diff_key] = diff_vol_flipped
        println("    ✅ Saved diffusivity for TP $tp_i")
        
        # Free memory
        ct_vol = nothing; pet_vol = nothing; ts_vol = nothing
        ct_native = nothing; pet_native = nothing; ts_native = nothing
        edge_map = nothing; diff_vol = nothing; diff_vol_flipped = nothing
        GC.gc(true)
        CUDA.reclaim()
    end
    
    close(h5_file)
    
    println("\n[3/3] Verification...")
    h5_file = h5open(h5_path, "r")
    tp_groups = get_tp_groups(h5_file)
    all_ok = true
    for (tp_idx, group_name) in enumerate(tp_groups)
        diff_key = "$group_name/diffusivity"
        if haskey(h5_file, diff_key)
            sz = size(read(h5_file[diff_key]))
            println("  ✅ TP $(tp_idx-1) ($group_name): diffusivity $sz")
        else
            println("  ❌ TP $(tp_idx-1) ($group_name): diffusivity MISSING")
            all_ok = false
        end
    end
    close(h5_file)
    
    if all_ok
        println("\n✅ All timepoints have diffusivity computed and stored in HDF5.")
    else
        println("\n⚠️  Some timepoints are missing diffusivity. Re-run this script.")
    end
end

main()
