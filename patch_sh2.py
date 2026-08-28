with open("/mnt/big/project_ssd/project_ssd/MedEye3d.jl/scripts/app/run_interactive_mrb.sh", "r") as f:
    content = f.read()

content = content.replace(
    'cd "$(dirname "$0")/.."',
    'cd "$(dirname "$0")/../.."'
)

with open("/mnt/big/project_ssd/project_ssd/MedEye3d.jl/scripts/app/run_interactive_mrb.sh", "w") as f:
    f.write(content)
