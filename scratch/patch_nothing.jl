content = read("src/display/LesionMetadataWindow.jl", String)
content = replace(content, "existing_loc = get(data, \"Anatomic Location\", \"\")" => "existing_loc = get(data, \"Anatomic Location\", \"\")\n        if existing_loc === nothing; existing_loc = \"\"; end")
content = replace(content, "existing_subloc = get(data, \"Anatomical Sublocation\", \"\")" => "existing_subloc = get(data, \"Anatomical Sublocation\", \"\")\n        if existing_subloc === nothing; existing_subloc = \"\"; end")
write("src/display/LesionMetadataWindow.jl", content)
