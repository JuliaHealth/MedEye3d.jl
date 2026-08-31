module AIInference

using MedImages
using JSON

export run_helpnet_inference, run_skellytour_segmentation, run_bone_subsegmentation, run_nninteractive_inference

# Resolve pyenv relative to the project root (works inside Docker and on host)
const _PROJECT_ROOT = abspath(joinpath(@__DIR__, "..", ".."))
const DEFAULT_PYENV = let
    pyenv_path = joinpath(_PROJECT_ROOT, "pyenv", "bin", "python3")
    isfile(pyenv_path) ? pyenv_path : "python3"  # fallback to system python
end
const HELPNET_BUNDLE_DIR = let
    # Try container path first, then host path
    for p in ["/workspaces/MedEye3d.jl/helpnet_inference_bundle",
              "/mnt/big/project_ssd/project_ssd/slicer_lesion_text_extension/src/helpnet_inference_bundle"]
        isdir(p) && return p
    end
    "/mnt/big/project_ssd/project_ssd/slicer_lesion_text_extension/src/helpnet_inference_bundle"
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

end # module AIInference
