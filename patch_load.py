import re

with open("/mnt/big/project_ssd/project_ssd/MedEye3d.jl/scripts/app/run_interactive_mrb.jl", "r") as f:
    content = f.read()

# Replace the anat_f32 logic
content = content.replace(
    'anat_f32 = e.anatomy !== nothing ? Float32.(e.anatomy) : zeros(Float32, size(e.ct))',
    'anat_f32 = !isempty(MEH.global_ts_atlas[]) ? Float32.(MEH.global_ts_atlas[]) : zeros(Float32, size(e.ct))'
)

# Strip out the per-TP anatomy loading block
start_str = "# Load per-TP max_anatomy atlas (prefer pre-registered/resampled volume from HDF5)"
end_str = "t_total_ms = (time_ns() - t_total) / 1e6"

start_idx = content.find(start_str)
end_idx = content.find(end_str)

if start_idx != -1 and end_idx != -1:
    content = content[:start_idx] + "anatomy_vol = nothing\n            " + content[end_idx:]
    with open("/mnt/big/project_ssd/project_ssd/MedEye3d.jl/scripts/app/run_interactive_mrb.jl", "w") as f:
        f.write(content)
    print("Successfully patched run_interactive_mrb.jl")
else:
    print("Could not find start or end index!")

