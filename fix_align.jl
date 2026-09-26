path = "src/display/LesionMetadataWindow.jl"
content = read(path, String)
content = replace(content, "halign=:fill, valign=:fill" => "")
write(path, content)
println("Fixed alignment issue.")
