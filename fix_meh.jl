path = "src/display/reactingToMouseKeyboard/ReactOnMouseClickAndDrag.jl"
content = read(path, String)
content = replace(content, "MEH = parentmodule(parentmodule(@__MODULE__)).SegmentationDisplay.MakieEventHandlers" => "MEH = parentmodule(@__MODULE__).SegmentationDisplay.MakieEventHandlers")
write(path, content)
println("Fixed MEH resolution")
