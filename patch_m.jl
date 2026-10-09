content = read("src/display/GLFW/MakieEventHandlers.jl", String)
content = replace(content, "ReactToScroll.reactToScrollMultiPanel!(collect(1:length(stateObjects)), stateObjects, \n            Dict(i => stateObjects[i].valueForMasToSet.textDispCoords for i in 1:length(stateObjects)))" => "ReactToScroll.reactToScrollMultiPanel!(collect(1:length(stateObjects)), stateObjects)")
write("src/display/GLFW/MakieEventHandlers.jl", content)
