path = "src/structs/Measurements.jl"
content = read(path, String)
content = replace(content, "length_mm::Float32 = 0.0f0     # Euclidean distance in mm\n    is_active::Bool = false" => "length_mm::Float32 = 0.0f0     # Euclidean distance in mm\n    suv_mean::Float32 = 0.0f0\n    suv_max::Float32 = 0.0f0\n    is_active::Bool = false")
write(path, content)
println("Patched LineMeasurement struct")
