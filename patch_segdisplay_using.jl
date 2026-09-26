path = "src/display/GLFW/SegmentationDisplay.jl"
content = read(path, String)
content = replace(content, "using ..ReactingToInput, ..ReactToScroll, ..DataStructs, ..StructsManag" => "using ..ReactingToInput, ..ReactToScroll, ..DataStructs, ..StructsManag, ..Measurements")
write(path, content)
