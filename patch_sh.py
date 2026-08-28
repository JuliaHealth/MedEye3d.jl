with open("/mnt/big/project_ssd/project_ssd/MedEye3d.jl/scripts/app/run_interactive_mrb.sh", "r") as f:
    content = f.read()

# Replace the run line
content = content.replace(
    'JULIA_NUM_THREADS=3,1 julia --project=. scripts/run_interactive_mrb.jl',
    'JULIA_NUM_THREADS=3,1 julia --project=. scripts/run_interactive_mrb.jl 2>&1 | tee app_execution.log'
)

with open("/mnt/big/project_ssd/project_ssd/MedEye3d.jl/scripts/app/run_interactive_mrb.sh", "w") as f:
    f.write(content)
print("Successfully patched run_interactive_mrb.sh")
