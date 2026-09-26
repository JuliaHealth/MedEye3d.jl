path = "src/structs/Measurements.jl"
content = read(path, String)

content = replace(content, "color = m.is_active ? Float32[1.0, 0.0, 0.0, 1.0] : Float32[1.0, 0.0, 1.0, 1.0]" => "color = m.is_active ? Float32[1.0, 0.0, 0.0, 0.4] : Float32[1.0, 0.0, 1.0, 0.4]")

content = replace(content, "color = lm.is_active ? Float32[1.0, 0.0, 0.0, 1.0] : Float32[1.0, 0.0, 1.0, 1.0]" => "color = lm.is_active ? Float32[1.0, 0.0, 0.0, 0.4] : Float32[1.0, 0.0, 1.0, 0.4]")

write(path, content)
