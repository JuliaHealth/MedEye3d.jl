with open("/mnt/big/project_ssd/project_ssd/MedEye3d.jl/src/display/LesionMetadataWindow.jl", "r") as f:
    content = f.read()

content = content.replace(
    'for sec in (sec_meta, sec_seg, sec_report, sec_radlex, sec_custom)',
    'for sec in (sec_meta, sec_seg, sec_report)'
)

with open("/mnt/big/project_ssd/project_ssd/MedEye3d.jl/src/display/LesionMetadataWindow.jl", "w") as f:
    f.write(content)
print("Successfully patched CV callback")
