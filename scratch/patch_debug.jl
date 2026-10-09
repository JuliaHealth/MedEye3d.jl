content = read("src/display/LesionMetadataWindow.jl", String)
old = """
            target_loc = anat_loc
            if (isempty(strip(existing_loc)) || existing_loc in ["N/A", "None", "Unknown"]) && !isempty(target_loc)
                data["Anatomic Location"] = target_loc
                db_updates["Anatomic Location"] = target_loc
                @debug "[PREFILL] Auto-filled Anatomic Location='\$target_loc'"
"""
new = """
            target_loc = anat_loc
            if (isempty(strip(existing_loc)) || existing_loc in ["N/A", "None", "Unknown"]) && !isempty(target_loc)
                data["Anatomic Location"] = target_loc
                db_updates["Anatomic Location"] = target_loc
                println("[DEBUG-PREFILL] existing_loc was empty/NA. Set target_loc='\$target_loc'")
"""
content = replace(content, old => new)
write("src/display/LesionMetadataWindow.jl", content)
