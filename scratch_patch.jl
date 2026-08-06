function updateQuadVertices!(stateObject::StateDataFields, layout::Symbol)
    calcDimStruct = stateObject.calcDimsStruct
    wc = calcDimStruct.widthCorr / 2.0f0
    hc = calcDimStruct.heightCorr / 2.0f0
    
    res = copy(calcDimStruct.mainImageQuadVert)
    
    if layout == :Hidden
        res[1], res[9], res[17], res[25] = 0.0f0, 0.0f0, 0.0f0, 0.0f0
        res[2], res[10], res[18], res[26] = 0.0f0, 0.0f0, 0.0f0, 0.0f0
    else
        # Y coordinates
        if layout == :TopLeft || layout == :TopRight
            res[2] = (1.0f0 - hc)
            res[10] = (0.0f0 + hc)
            res[18] = (0.0f0 + hc)
            res[26] = (1.0f0 - hc)
        elseif layout == :BottomLeft || layout == :BottomRight
            res[2] = (0.0f0 - hc)
            res[10] = (-1.0f0 + hc)
            res[18] = (-1.0f0 + hc)
            res[26] = (0.0f0 - hc)
        elseif layout == :LeftHalf || layout == :RightHalf
            res[2] = (1.0f0 - hc)
            res[10] = (-1.0f0 + hc)
            res[18] = (-1.0f0 + hc)
            res[26] = (1.0f0 - hc)
        end
        
        # X coordinates
        if layout == :TopLeft || layout == :BottomLeft
            res[1] = (0.0f0 - wc)
            res[9] = (0.0f0 - wc)
            res[17] = (-1.0f0 + wc)
            res[25] = (-1.0f0 + wc)
        elseif layout == :TopRight || layout == :BottomRight
            res[1] = (1.0f0 - wc)
            res[9] = (1.0f0 - wc)
            res[17] = (0.0f0 + wc)
            res[25] = (0.0f0 + wc)
        elseif layout == :LeftHalf
            res[1] = (0.0f0 - wc)
            res[9] = (0.0f0 - wc)
            res[17] = (-1.0f0 + wc)
            res[25] = (-1.0f0 + wc)
        elseif layout == :RightHalf
            res[1] = (1.0f0 - wc)
            res[9] = (1.0f0 - wc)
            res[17] = (0.0f0 + wc)
            res[25] = (0.0f0 + wc)
        end
    end
    
    stateObject.calcDimsStruct = Setfield.setproperties(calcDimStruct, (mainImageQuadVert = res,))
    
    ModernGL.glBindBuffer(ModernGL.GL_ARRAY_BUFFER, stateObject.mainForDisplayObjects.vbo)
    ModernGL.glBufferData(ModernGL.GL_ARRAY_BUFFER, sizeof(res), res, ModernGL.GL_STATIC_DRAW)
end
