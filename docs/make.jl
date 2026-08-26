using Documenter, DocumenterVitepress

makedocs(; 
    sitename = "MedEye3d.jl", 
    authors = "Jakub Mitura <jakub.mitura14@gmail>, Beata E. Chrapko and Divyansh Goyal <divital2004@gmail.com>",
    format=DocumenterVitepress.MarkdownVitepress(
        repo = "github.com/JuliaHealth/MedEye3d.jl", 
        devbranch = "master",
        devurl = "dev",
    ),
    warnonly = true,
    draft = false,
    source = "src",
    build = "build",
    pages=[
        "Manual" => [
            "Get Started" => "manual/get_started.md",
            "Code Examples" => "manual/code_example.md",
            "AI Inference Pipeline" => "manual/ai_inference_pipeline.md",
            "GPU Kernels & Post-Processing" => "manual/gpu_kernels_and_postprocessing.md",
            "QuadView & 3D Navigation" => "manual/quad_view_and_navigation.md",
            "Bone Subsegmentation & MRB" => "manual/bone_subsegmentation_and_mrb.md",
            "GUI Controls & Windowing" => "manual/gui_controls_and_windowing.md",
            "Lesion Metadata & Tracking" => "manual/lesion_metadata_and_tracking.md"
        ],
        "Developers' documentation" => [
            "Visualization Playbook" => "devs/playbook.md"
        ],
        "api" => "api.md"
        ],
)

# This is the critical part that creates the version structure
DocumenterVitepress.deploydocs(;
    repo = "github.com/JuliaHealth/MedImages.jl", 
    devbranch = "master",
    push_preview = true,
)
