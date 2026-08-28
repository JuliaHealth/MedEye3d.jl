import re

with open("/mnt/big/project_ssd/project_ssd/MedEye3d.jl/src/display/LesionMetadataWindow.jl", "r") as f:
    content = f.read()

# Extract the RadLex block
radlex_pattern = re.compile(r'    # ── RadLex Multi-Value Panel ───.*?\n.*?end_section!\(sec_radlex\)\n', re.DOTALL)
radlex_match = radlex_pattern.search(content)
radlex_code = radlex_match.group(0)
content = content[:radlex_match.start()] + content[radlex_match.end():]

# Extract the Custom block
custom_pattern = re.compile(r'    # ── Custom Key-Value Fields ───.*?\n.*?end_section!\(sec_custom\)\n', re.DOTALL)
custom_match = custom_pattern.search(content)
custom_code = custom_match.group(0)
content = content[:custom_match.start()] + content[custom_match.end():]

# Clean the extracted blocks
radlex_code = radlex_code.replace('sec_radlex = begin_section!("RadLex Ontology Properties")', 'Label(g[nr!(), 1:4], "-- RadLex Ontology Properties --", fontsize = 11, color = ACCENT, halign = :center, tellwidth = false)')
radlex_code = radlex_code.replace('end_section!(sec_radlex)', '')

custom_code = custom_code.replace('sec_custom = begin_section!("Custom Key-Value Fields")', 'Label(g[nr!(), 1:4], "-- Custom Key-Value Fields --", fontsize = 11, color = ACCENT, halign = :center, tellwidth = false)')
custom_code = custom_code.replace('end_section!(sec_custom)', '')

# Wait! We need to dynamically render the custom fields, because otherwise they are invisible!
# If `custom_db` changes, we need to show the key/value pairs.
# Actually, the user didn't ask for a fix for them not displaying, but it makes sense to render them.
# I will just move them for now as requested.

combined_code = radlex_code + "\n" + custom_code + "\n"

# Find the insertion point: end_section!(sec_meta)
insert_target = '    end_section!(sec_meta)'
insert_idx = content.find(insert_target)

if insert_idx != -1:
    content = content[:insert_idx] + combined_code + content[insert_idx:]
    with open("/mnt/big/project_ssd/project_ssd/MedEye3d.jl/src/display/LesionMetadataWindow.jl", "w") as f:
        f.write(content)
    print("Successfully patched LesionMetadataWindow.jl")
else:
    print("Could not find insertion point!")

