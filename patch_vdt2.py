with open("/mnt/big/project_ssd/project_ssd/MedEye3d.jl/scripts/app/run_interactive_mrb.jl", "r") as f:
    content = f.read()

content = content.replace(
    'anat_f32 = !isempty(MEH.global_ts_atlas[]) ? Float32.(MEH.global_ts_atlas[]) : zeros(Float32, size(e.ct))',
    'anat_f32 = MEH.global_ts_atlas[] !== nothing ? Float32.(MEH.global_ts_atlas[]) : zeros(Float32, size(e.ct))'
)

with open("/mnt/big/project_ssd/project_ssd/MedEye3d.jl/scripts/app/run_interactive_mrb.jl", "w") as f:
    f.write(content)
