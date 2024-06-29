"""
controls mask visibility responds to keyboard input
"""
module KeyboardVisibility
using ModernGL, ..DisplayWords, ..StructsManag, Setfield, ..PrepareWindow, ..DataStructs, Rocket, GLFW, Dictionaries, ..ForDisplayStructs, ..TextureManag, ..OpenGLDisplayUtils, ..Uniforms, Match, Parameters, DataTypesBasic
export processKeysInfo, setVisAndRender

"""
processing information from keys - the instance of this function will be chosen on
the basis mainly of multiple dispatch
"""
function processKeysInfo(textSpecObs::Identity{TextureSpec{T}}, stateObject::StateDataFields, keyInfo::KeyboardStruct) where {T}
    textSpec = textSpecObs.value
    if (keyInfo.isCtrlPressed)
        setVisAndRender(false, stateObject, textSpec.uniforms)
        @info " set visibility of $(textSpec.name) to false"
        #to enabling undoing it
        # addToforUndoVector(stateObject, ()-> setVisAndRender(true,stateObject,textSpec.uniforms )    )
    elseif (keyInfo.isShiftPressed)
        setVisAndRender(true, stateObject, textSpec.uniforms)
        @info " set visibility of $(textSpec.name) to true"
        #to enabling undoing it
        #    addToforUndoVector(stateObject, ()->setVisAndRender(false,stateObject,textSpec.uniforms )   )

    elseif (keyInfo.isAltPressed)
        oldTex = stateObject.textureToModifyVec
        stateObject.textureToModifyVec = [textSpec]
        @info " set texture for manual modifications to  $(textSpec.name)"
        if (!isempty(oldTex))
            #    addToforUndoVector(stateObject, ()->begin  @info stateObject.textureToModifyVec=[oldTex[1]] end)
        end
    end #if
end #processKeysInfo

"""
sets  visibility and render the result to the screen
"""
function setVisAndRender(isVis::Bool, stateObject::StateDataFields, unifs::TextureUniforms)
    setTextureVisibility(isVis, unifs)
    basicRender(stateObject.mainForDisplayObjects.window)

end#setVisAndRender

end#module
