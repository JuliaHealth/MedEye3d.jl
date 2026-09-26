path = "src/structs/Measurements.jl"
content = read(path, String)
content = replace(content, "t = 0.003f0" => "t = 0.015f0")
content = replace(content, "color = m.is_active ? Float32[0.0, 1.0, 0.0, 1.0] : Float32[1.0, 1.0, 0.0, 1.0]" => "color = m.is_active ? Float32[1.0, 0.0, 0.0, 1.0] : Float32[1.0, 0.0, 1.0, 1.0]")
content = replace(content, "color = lm.is_active ? Float32[0.0, 1.0, 0.0, 1.0] : Float32[1.0, 1.0, 0.0, 1.0]" => "color = lm.is_active ? Float32[1.0, 0.0, 0.0, 1.0] : Float32[1.0, 0.0, 1.0, 1.0]")
write(path, content)
