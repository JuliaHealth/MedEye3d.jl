"""
controls mask visibility responds to keyboard input
"""
module KeyboardVisibility
using ModernGL, ..DisplayWords, ..StructsManag, Setfield, ..PrepareWindow,   ..DataStructs ,Glutils, Rocket, GLFW,Dictionaries,  ..ForDisplayStructs, ..TextureManag,  ..OpenGLDisplayUtils,  ..Uniforms, Match, Parameters,DataTypesBasic
export processKeysInfo, setVisAndRender   

"""
processing information from keys - the instance of this function will be chosen on
the basis mainly of multiple dispatch
"""
function processKeysInfo(textSpecObs::Identity{TextureSpec{T}},actor::SyncActor{Any, ActorWithOpenGlObjects},keyInfo::KeyboardStruct )where T
    textSpec =  textSpecObs.value
    if(keyInfo.isCtrlPressed)    
        setVisAndRender(false,actor.actor,textSpec.uniforms )
        @info " set visibility of $(textSpec.name) to false" 
        #to enabling undoing it 
        addToforUndoVector(actor, ()-> setVisAndRender(true,actor.actor,textSpec.uniforms )    )
    elseif(keyInfo.isShiftPressed)  
        setVisAndRender(true,actor.actor,textSpec.uniforms )
        @info " set visibility of $(textSpec.name) to true" 
       #to enabling undoing it 
       addToforUndoVector(actor, ()->setVisAndRender(false,actor.actor,textSpec.uniforms )   )

    elseif(keyInfo.isAltPressed)  
        oldTex = actor.actor.textureToModifyVec
        actor.actor.textureToModifyVec= [textSpec]
        @info " set texture for manual modifications to  $(textSpec.name)"
       if(!isempty(oldTex))
       addToforUndoVector(actor, ()->begin  @info actor.actor.textureToModifyVec=[oldTex[1]] end)
       end
    end #if
end #processKeysInfo

"""
sets  visibility and render the result to the screen
"""
function setVisAndRender(isVis::Bool,actor::ActorWithOpenGlObjects,unifs::TextureUniforms )
    setTextureVisibility(isVis,unifs )
    basicRender(actor.mainForDisplayObjects.window)

end#setVisAndRender

end#module