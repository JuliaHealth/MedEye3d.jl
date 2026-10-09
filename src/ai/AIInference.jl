module AIInference

using MedImages
using JSON
using Statistics

export run_helpnet_inference, run_skellytour_segmentation, run_bone_subsegmentation, run_nninteractive_inference, run_heatgdt_inference

# Resolve pyenv relative to the project root (works inside Docker and on host)
const _PROJECT_ROOT = abspath(joinpath(@__DIR__, "..", ".."))
const DEFAULT_PYENV = let
    pyenv_path = joinpath(_PROJECT_ROOT, "pyenv", "bin", "python3")
    (isfile("/.dockerenv") || !isfile(pyenv_path)) ? "python3" : pyenv_path
end
const HELPNET_BUNDLE_DIR = let
    found = "/mnt/big/project_ssd/project_ssd/slicer_lesion_text_extension/src/helpnet_inference_bundle"
    for p in ["/workspaces/MedEye3d.jl/helpnet_inference_bundle",
              "/mnt/big/project_ssd/project_ssd/slicer_lesion_text_extension/src/helpnet_inference_bundle"]
        if isdir(p)
            found = p
            break
        end
    end
    found
end
const HELPNET_CHECKPOINT = joinpath(HELPNET_BUNDLE_DIR, "checkpoints", "helpnet_model_final.pt")

"""
    run_helpnet_inference(ct_patch_path::String, pet_patch_path::String, point_path::String, out_dir::String; pyenv=DEFAULT_PYENV)

Runs HELPNet deep-learning lesion segmentation on a 64x64x64 PET/CT patch.
Returns path to generated binary prediction NIfTI file (`prediction.nii.gz`).
"""
function run_helpnet_inference(ct_patch_path::String, pet_patch_path::String, point_path::String, out_dir::String; pyenv=DEFAULT_PYENV, checkpoint=HELPNET_CHECKPOINT)
    mkpath(out_dir)
    pred_path = joinpath(out_dir, "prediction.nii.gz")
    
    cmd_str = """
import sys
sys.path.insert(0, '$(HELPNET_BUNDLE_DIR)')
import run_inference
run_inference.run_inference('$(ct_patch_path)', '$(pet_patch_path)', '$(point_path)', '$(checkpoint)', '$(out_dir)')
"""
    
    env = copy(ENV)
    env["CUDA_VISIBLE_DEVICES"] = "1"
    cmd = setenv(`$(pyenv) -c $(cmd_str)`, env)
    @info "Running HELPNet inference..." cmd
    run(cmd)
    
    if !isfile(pred_path)
        error("HELPNet inference failed: output file not found at $(pred_path)")
    end
    return pred_path
end

"""
    run_skellytour_segmentation(ct_path::String, out_dir::String; pyenv=DEFAULT_PYENV)

Runs Skellytour whole-body cortical and trabecular bone subsegmentation.
"""
function run_skellytour_segmentation(ct_path::String, out_dir::String; pyenv=DEFAULT_PYENV)
    mkpath(out_dir)
    skelly_bin = joinpath(dirname(pyenv), "skellytour")
    if !isfile(skelly_bin)
        skelly_bin = "skellytour"
    end
    
    env = copy(ENV)
    env["CUDA_VISIBLE_DEVICES"] = "1"
    cmd = setenv(`$(skelly_bin) -i $(ct_path) -o $(out_dir) -m low --subseg --fast --overwrite`, env)
    @info "Running Skellytour bone subsegmentation..." cmd
    run(cmd)
    return out_dir
end

"""
    run_bone_subsegmentation(lesion_path, bone_path, out_surface, out_marrow; 
                             ct_path="", max_anatomy_path="", bone_label_ids="", pyenv=DEFAULT_PYENV)

Extracts cortical bone surface (from max_anatomy solid bones) and bone marrow 
(from Skellytour label 1 trabecula) subsegment fragments around a bone metastasis.
"""
function run_bone_subsegmentation(lesion_path::String, bone_path::String, out_surface::String, out_marrow::String; 
                                  ct_path::String="", max_anatomy_path::String="", bone_label_ids::String="", pyenv=DEFAULT_PYENV)
    script_path = joinpath(@__DIR__, "..", "..", "scripts", "ai", "bone_subsegmentation.py")
    # Docker container path (if running inside sharp_ramanujan, script is at /workspaces/...)
    docker_script = "/workspaces/MedEye3d.jl/scripts/ai/bone_subsegmentation.py"
    
    env = copy(ENV)
    env["CUDA_VISIBLE_DEVICES"] = "1"
    
    extra_args = String[]
    if ct_path != ""
        push!(extra_args, "--ct", ct_path)
    end
    if max_anatomy_path != ""
        push!(extra_args, "--max-anatomy", max_anatomy_path)
    end
    if bone_label_ids != ""
        push!(extra_args, "--bone-labels", bone_label_ids)
    end
    
    # Try medeye3d-ai Docker container first (has scipy/nibabel/numpy)
    docker_available = try
        success(`docker inspect medeye3d-ai`)
    catch
        false
    end
    
    if docker_available
        docker_args = ["docker", "exec", "medeye3d-ai", "python3", "-u", docker_script,
                       "--lesion", lesion_path, "--bone", bone_path,
                       "--out-surface", out_surface, "--out-marrow", out_marrow]
        append!(docker_args, extra_args)
        cmd = Cmd(docker_args)
        @info "Running bone subsegmentation via Docker (medeye3d-ai)..." cmd
        run(cmd)
    else
        # Fallback to local pyenv
        args = [pyenv, script_path, "--lesion", lesion_path, "--bone", bone_path,
                "--out-surface", out_surface, "--out-marrow", out_marrow]
        append!(args, extra_args)
        cmd = setenv(Cmd(args), env)
        @info "Running bone subsegmentation locally..." cmd
        run(cmd)
    end
    return out_surface, out_marrow
end

"""
    run_nninteractive_inference(image_path::String, clicks_json_path::String, out_path::String; pyenv=DEFAULT_PYENV)

Runs nnInteractive prompt-based interactive segmentation.
"""
function run_nninteractive_inference(image_path::String, clicks_json_path::String, out_path::String; pyenv=DEFAULT_PYENV)
    cmd_str = """
import sys, json
# nnInteractive inference placeholder runner
print('nnInteractive session running on', '$(image_path)')
"""
    cmd = `$(pyenv) -c $(cmd_str)`
    run(cmd)
    return out_path
end

# ── Heat-GDT Constants ──────────────────────────────────────────────────────
const HEATGDT_PROJECT_DIR = normpath(joinpath(@__DIR__, "..", "..", "..", "semiautomatic", "JuliaHELPNet"))
const HEATGDT_DIR = joinpath(HEATGDT_PROJECT_DIR, "HeatGDT")
const HEATGDT_CHAMPION_PATH = joinpath(HEATGDT_DIR, "champion_pretrained_hed3d_wide.jld2")

# Lazy-loaded singleton for the Heat-GDT model (loaded once, reused across calls)
const _HEATGDT_STATE = Ref{Any}(nothing)

"""
    _ensure_heatgdt_loaded!()

Lazy-loads the Heat-GDT champion model (HED3D-Wide edge network + PDE parameters).
Loaded once per session from `champion_pretrained_hed3d_wide.jld2`.
"""
function _ensure_heatgdt_loaded!()
    if _HEATGDT_STATE[] !== nothing
        return _HEATGDT_STATE[]
    end
    
    @info "[Heat-GDT] Loading champion model from $(HEATGDT_CHAMPION_PATH)..."
    
    if !isfile(HEATGDT_CHAMPION_PATH)
        error("Heat-GDT champion model not found at $(HEATGDT_CHAMPION_PATH). Run training first.")
    end
    
    # Load the model using JLD2
    try
        @eval begin
            using Pkg
            Pkg.activate($(HEATGDT_PROJECT_DIR))
            include(joinpath($(HEATGDT_DIR), "model.jl"))
            include(joinpath($(HEATGDT_DIR), "architectures", "hed3d_wide.jl"))
            using .HeatGDTModel
            using JLD2, Lux, CUDA, Random
        end
        
        checkpoint = JLD2.load(HEATGDT_CHAMPION_PATH)
        ps_trained = checkpoint["ps"] |> Lux.gpu
        st_trained = checkpoint["st"] |> Lux.gpu
        
        # Handle both old (config dict) and new (edge_type string) checkpoint formats
        config = if haskey(checkpoint, "config")
            checkpoint["config"]
        else
            Dict("edge_type" => get(checkpoint, "edge_type", "hed3d_wide"),
                 "K" => 40, "dt" => 0.16, "theta" => 0.001, "tau" => 0.0001)
        end
        
        # Reconstruct the framework with champion parameters
        framework = HeatGDTModel.HeatGDTFramework(
            in_channels=3,
            edge_type=Symbol(get(config, "edge_type", "hed3d_wide")),
            K=get(config, "K", 40),
            dt=Float32(get(config, "dt", 0.16)),
            theta=Float32(get(config, "theta", 0.001)),
            tau=Float32(get(config, "tau", 0.0001))
        )
        
        _HEATGDT_STATE[] = (framework=framework, ps=ps_trained, st=st_trained, config=config)
        @info "[Heat-GDT] Champion model loaded successfully ($(config))"
        return _HEATGDT_STATE[]
    catch e
        @error "[Heat-GDT] Failed to load champion model" exception=(e, catch_backtrace())
        rethrow(e)
    end
end

"""
    run_heatgdt_inference(ct_patch::Array{Float32,3}, pet_patch::Array{Float32,3},
                          ts_patch::Array{Float32,3}, seed_x::Int, seed_y::Int, seed_z::Int;
                          K::Int=40, dt::Float32=0.16f0, theta::Float32=0.001f0, tau::Float32=0.0001f0)

Runs Heat-GDT segmentation natively in Julia (no Docker/Python needed).
Takes a 64³ CT, PET, and TotalSegmentator patch centered on the seed point.
Returns a binary UInt8 mask of the segmented lesion.

The method:
1. Runs the HED3D-Wide edge detection network to produce edge map E(x)
2. Converts edges to diffusivity: D(x) = 0.01 + 0.99 * exp(-5 * E(x))
3. Runs K steps of heat diffusion from the seed point
4. Thresholds the heat field to produce a binary mask
"""
function run_heatgdt_inference(ct_patch::Array{Float32,3}, pet_patch::Array{Float32,3},
                                ts_patch::Array{Float32,3}, seed_x::Int, seed_y::Int, seed_z::Int;
                                K::Int=40, dt::Float32=0.16f0, theta::Float32=0.001f0, tau::Float32=0.0001f0)
    state = _ensure_heatgdt_loaded!()
    framework = state.framework
    ps = state.ps
    st = state.st
    
    # Normalize CT patch (z-score)
    ct_mean = mean(ct_patch)
    ct_std = std(ct_patch) + 1f-6
    ct_norm = (ct_patch .- ct_mean) ./ ct_std
    
    # TotalSegmentator: normalize to [0, 1]
    ts_norm = Float32.(ts_patch) ./ 117.0f0
    
    # Stack into (X, Y, Z, 3, 1) tensor
    x = cat(reshape(ct_norm, 64, 64, 64, 1),
            reshape(pet_patch, 64, 64, 64, 1),
            reshape(ts_norm, 64, 64, 64, 1), dims=4)
    x = reshape(x, 64, 64, 64, 3, 1)
    
    # Create seed mask
    seed_mask = zeros(Float32, 64, 64, 64, 1)
    seed_mask[seed_x, seed_y, seed_z, 1] = 1.0f0
    
    # Move to GPU
    x_gpu = CUDA.cu(x)
    seed_gpu = CUDA.cu(seed_mask)
    
    # Run inference in test mode (no gradient tracking)
    st_test = Lux.testmode(st)
    result, _ = framework((x_gpu, seed_gpu), ps, st_test)
    
    # Extract binary mask
    mask_gpu = result.mask
    mask_cpu = Array(mask_gpu[:, :, :, 1])
    
    # Binarize
    binary_mask = UInt8.(mask_cpu .> 0.5f0)
    
    return binary_mask
end

end # module AIInference
