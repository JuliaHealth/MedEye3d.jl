with open("/mnt/big/project_ssd/project_ssd/MedEye3d.jl/src/display/LesionMetadataWindow.jl", "r") as f:
    content = f.read()

content = content.replace(
    'if haskey(_MEH.tp_data_cache, tp) && !isempty(_MEH.global_ts_atlas[])',
    'if haskey(_MEH.tp_data_cache, tp) && _MEH.global_ts_atlas[] !== nothing'
)

with open("/mnt/big/project_ssd/project_ssd/MedEye3d.jl/src/display/LesionMetadataWindow.jl", "w") as f:
    f.write(content)
