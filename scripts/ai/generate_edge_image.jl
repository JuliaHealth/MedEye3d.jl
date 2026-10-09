using Pkg
Pkg.activate("/mnt/big/project_ssd/project_ssd/semiautomatic/JuliaHELPNet")

using Lux
using LuxCUDA
using ComponentArrays
using JLD2
using NNlib
using NIfTI
using Statistics
using CUDA

include("/mnt/big/project_ssd/project_ssd/semiautomatic/JuliaHELPNet/HeatGDT/architectures/hed3d_wide.jl")

# Device helpers — modern Lux uses gpu_device()/cpu_device()
const gdev = gpu_device()
const cdev = cpu_device()

function compute_diffusivity(edge_map, epsilon_D::Float32=0.01f0, gamma::Float32=5.0f0)
    e = edge_map
    return epsilon_D .+ (1.0f0 - epsilon_D) .* exp.(-gamma .* e)
end

function generate_edge_image(ct_path, pet_path, totalseg_path, output_path)
    println("Loading model...")
    model = HED3D_Wide(3)
    
    jld_path = "/mnt/big/project_ssd/project_ssd/semiautomatic/JuliaHELPNet/HeatGDT/champion_pretrained_hed3d_wide.jld2"
    ckpt = JLD2.load(jld_path)
    ps_full = ComponentArray(ckpt["ps"]) |> gdev
    # The saved checkpoint wraps params under 'edge_net' key (from training wrapper).
    # Unwrap to match HED3D's expected parameter layout (e1, e2, ..., fuse).
    ps = ps_full.edge_net
    st_full = Lux.testmode(ckpt["st"]) |> gdev
    st = st_full.edge_net

    println("Loading volumes...")
    ct_nii = niread(ct_path)
    pet_nii = niread(pet_path)
    ts_nii = niread(totalseg_path)

    ct = Float32.(ct_nii.raw)
    pet = Float32.(pet_nii.raw)
    ts = Float32.(ts_nii.raw)

    sz = size(ct)
    @assert sz == size(ts) "CT and TotalSeg shapes must match! Got CT=$(sz), TS=$(size(ts))"
    
    # Handle PET/CT resolution mismatch: resample PET to CT grid if needed
    if size(pet) != sz
        println("  PET resolution $(size(pet)) differs from CT $(sz), resampling PET to CT grid...")
        # Use NNlib trilinear interpolation: input must be (X, Y, Z, C, B)
        pet_5d = reshape(pet, size(pet)..., 1, 1)
        target_size = (sz[1], sz[2], sz[3])
        pet_resampled = NNlib.upsample_trilinear(pet_5d; size=target_size)
        pet = dropdims(pet_resampled, dims=(4, 5))
        println("  PET resampled to $(size(pet))")
    end

    patch_size = (64, 64, 64)
    stride = (32, 32, 32)

    edge_volume = zeros(Float32, sz)
    weight_volume = zeros(Float32, sz)

    println("Running sliding window inference...")
    
    # Pre-allocate CPU input buffer
    input_buf = Array{Float32}(undef, 64, 64, 64, 3, 1)
    patch_count = 0
    total_patches = length(1:stride[3]:sz[3]) * length(1:stride[2]:sz[2]) * length(1:stride[1]:sz[1])
    t_start = time()
    
    for z in 1:stride[3]:sz[3]
        for y in 1:stride[2]:sz[2]
            for x in 1:stride[1]:sz[1]
                x1, y1, z1 = x, y, z
                x2 = min(x1 + patch_size[1] - 1, sz[1])
                y2 = min(y1 + patch_size[2] - 1, sz[2])
                z2 = min(z1 + patch_size[3] - 1, sz[3])
                
                if x2 - x1 + 1 < patch_size[1]
                    x1 = max(1, x2 - patch_size[1] + 1)
                end
                if y2 - y1 + 1 < patch_size[2]
                    y1 = max(1, y2 - patch_size[2] + 1)
                end
                if z2 - z1 + 1 < patch_size[3]
                    z1 = max(1, z2 - patch_size[3] + 1)
                end

                ct_patch = @view ct[x1:x2, y1:y2, z1:z2]
                pet_patch = @view pet[x1:x2, y1:y2, z1:z2]
                ts_patch = @view ts[x1:x2, y1:y2, z1:z2]

                # z-score normalization per patch
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
                
                # Light GC every 200 patches to prevent OOM
                if patch_count % 200 == 0
                    GC.gc(false)  # non-full GC
                end
            end
        end
    end

    # Average overlapping regions
    edge_volume ./= max.(weight_volume, 1.0f0)

    # Compute diffusivity: D(x) = 0.01 + 0.99 * exp(-5.0 * E(x))
    diff_volume = compute_diffusivity(edge_volume)

    println("Saving outputs...")
    edge_out_path = output_path * "_edge.nii.gz"
    diff_out_path = output_path * "_diffusivity.nii.gz"

    niwrite(edge_out_path, NIVolume(ct_nii.header, edge_volume))
    niwrite(diff_out_path, NIVolume(ct_nii.header, diff_volume))
    
    println("Done!")
end

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) < 4
        println("Usage: julia generate_edge_image.jl <ct_path> <pet_path> <totalseg_path> <output_prefix>")
        exit(1)
    end
    generate_edge_image(ARGS[1], ARGS[2], ARGS[3], ARGS[4])
end
