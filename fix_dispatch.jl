path = "src/display/GLFW/SegmentationDisplay.jl"
content = read(path, String)
code = """
on_next!(stateObjects::Vector{StateDataFields}, data::MakieEvents.SetTPLastEvent) = MakieEventHandlers.reactToSetTPLast(data, stateObjects)

on_next!(stateObjects::Vector{StateDataFields}, data::MakieEvents.JumpToMeasurementEvent) = MakieEventHandlers.reactToJumpToMeasurement(data, stateObjects)
on_next!(stateObjects::Vector{StateDataFields}, data::MakieEvents.JumpToLineMeasurementEvent) = MakieEventHandlers.reactToJumpToLineMeasurement(data, stateObjects)
on_next!(stateObjects::Vector{StateDataFields}, data::MakieEvents.DeleteMeasurementEvent) = MakieEventHandlers.reactToDeleteMeasurement(data, stateObjects)
on_next!(stateObjects::Vector{StateDataFields}, data::MakieEvents.DeleteLineMeasurementEvent) = MakieEventHandlers.reactToDeleteLineMeasurement(data, stateObjects)
on_next!(stateObjects::Vector{StateDataFields}, data::MakieEvents.ToggleMeasurementModeEvent) = MakieEventHandlers.reactToToggleMeasurementMode(data, stateObjects)
on_next!(stateObjects::Vector{StateDataFields}, data::MakieEvents.CycleMeasurementSubModeEvent) = MakieEventHandlers.reactToCycleMeasurementSubMode(data, stateObjects)

on_error!(stateObjects::Vector{StateDataFields}, err) = error(err)
"""
content = replace(content, """
on_next!(stateObjects::Vector{StateDataFields}, data::MakieEvents.SetTPLastEvent) = MakieEventHandlers.reactToSetTPLast(data, stateObjects)
on_error!(stateObjects::Vector{StateDataFields}, err) = error(err)
""" => code)
write(path, content)
println("Patched event dispatches")
