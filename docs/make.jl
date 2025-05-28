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
            "Code" => "manual/code_example.md"
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
