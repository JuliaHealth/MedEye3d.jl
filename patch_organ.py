import re

with open('src/display/LesionMetadataWindow.jl', 'r') as f:
    content = f.read()

start_str = "        on(_MEH.organ_mapping_updated) do (lid, organ_name)\n            try\n                @debug"
end_str = "            catch e\n                @warn \"Failed to register organ_mapping_updated listener"

match_start = content.find("        on(_MEH.organ_mapping_updated) do (lid, organ_name)")

if match_start != -1:
    end_idx = content.find("            catch e", match_start)
    end_idx = content.find("        end", end_idx) + 11
    
    old_block = content[match_start:end_idx]
    print(old_block[-100:])
    
    new_block = old_block.replace("        on(_MEH.organ_mapping_updated) do (lid, organ_name)\n            try", "        on(_MEH.organ_mapping_updated) do (lid, organ_name)\n            lock(ui_lock) do\n                push!(ui_queue, () -> begin\n                    try")
    new_block = new_block.replace("            catch e\n                @warn \"Error processing organ_mapping_updated: $e\"\n            end\n        end", "                    catch e\n                        @warn \"Error processing organ_mapping_updated: $e\"\n                    end\n                end)\n            end\n        end")
    
    new_content = content[:match_start] + new_block + content[end_idx:]
    with open('src/display/LesionMetadataWindow.jl', 'w') as f:
        f.write(new_content)
    print("Patch applied.")
else:
    print("Could not find block.")
