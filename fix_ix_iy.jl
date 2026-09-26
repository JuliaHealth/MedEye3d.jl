path = "src/display/reactingToMouseKeyboard/ReactOnMouseClickAndDrag.jl"
content = read(path, String)
content = replace(content, """
    texX, texY = getTextureCoordinatesFromScreen(x, y, panelState.calcDimsStruct, actualW, actualH)
    w = Float64(panelState.calcDimsStruct.imageTextureWidth)
    h = Float64(panelState.calcDimsStruct.imageTextureHeight)
    
    ix = round(Int, texX * w)
    iy = round(Int, texY * h)
""" => """
    texX, texY = getTextureCoordinatesFromScreen(x, y, panelState.calcDimsStruct, actualW, actualH)
    w = Float64(panelState.calcDimsStruct.imageTextureWidth)
    h = Float64(panelState.calcDimsStruct.imageTextureHeight)
    
    ix = round(Int, texX)
    iy = round(Int, texY)
""")
write(path, content)
