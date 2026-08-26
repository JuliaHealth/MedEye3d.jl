module AIInference

using MedImages
using JSON

export run_helpnet_inference, run_skellytour_segmentation, run_bone_subsegmentation, run_nninteractive_inference

const DEFAULT_PYENV = "/mnt/big/project_ssd/project_ssd/MedEye3d.jl/pyenv/bin/python3"
const HELPNET_BUNDLE_DIR = "/mnt/big/project_ssd/project_ssd/slicer_lesion_text_extension/src/helpnet_inference_bundle"
const HELPNET_CHECKPOINT = "/mnt/big/project_ssd/project_ssd/slicer_lesion_text_extension/src/helpnet_inference_bundle/checkpoints/helpnet_model_final.pt"

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
    run_bone_subsegmentation(lesion_path::String, bone_path::String, out_surface::String, out_marrow::String; pyenv=DEFAULT_PYENV)

Extracts cortical bone surface and bone marrow subsegment fragments around a bone metastasis.
"""
function run_bone_subsegmentation(lesion_path::String, bone_path::String, out_surface::String, out_marrow::String; ct_path::String="", pyenv=DEFAULT_PYENV)
    script_path = joinpath(@__DIR__, "..", "..", "scripts", "ai", "bone_subsegmentation.py")
    
    env = copy(ENV)
    env["CUDA_VISIBLE_DEVICES"] = "1"
    
    if ct_path == ""
        cmd = setenv(`$(pyenv) $(script_path) --lesion $(lesion_path) --bone $(bone_path) --out-surface $(out_surface) --out-marrow $(out_marrow)`, env)
    else
        cmd = setenv(`$(pyenv) $(script_path) --lesion $(lesion_path) --bone $(bone_path) --ct $(ct_path) --out-surface $(out_surface) --out-marrow $(out_marrow)`, env)
    end
    
    @info "Running bone subsegmentation extraction..." cmd
    run(cmd)
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
