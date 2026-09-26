path = "src/structs/Measurements.jl"
content = read(path, String)
content = replace(content, "l.length_mm" => "lm.length_mm")
write(path, content)
println("Fixed line rendering bug.")
