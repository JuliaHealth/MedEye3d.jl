"""
functions to controll stroke width , setting which texture is currently active and actions undoing
"""
module OtherKeyboardActions
using ModernGL, ..DisplayWords, ..StructsManag, Setfield, ..PrepareWindow,   ..DataStructs , Rocket, GLFW,Dictionaries,  ..ForDisplayStructs, ..TextureManag,  ..OpenGLDisplayUtils,  ..Uniforms, Match, Parameters,DataTypesBasic   

"""
in case we want to  get new number set for manual modifications
    toBeSavedForBack - just marks weather we wat to save the info how to undo latest action
    - false if we invoke it from undoing 
"""
function processKeysInfo(numbb::Identity{Int64}
                        ,actor::SyncActor{Any, ActorWithOpenGlObjects}
                        ,keyInfo::KeyboardStruct 
                        ,toBeSavedForBack::Bool = true) where T

    @info "nnnnn $(numbb.value)"
    valueForMasToSett = valueForMasToSetStruct(value = numbb.value)
    old = actor.actor.valueForMasToSet.value
    actor.actor.valueForMasToSet =valueForMasToSett
    textureList= actor.actor.textureToModifyVec

    # in case we increase number it should not be outside of the possible values
    if(!isempty(textureList))
        @info max(Float32(textureList[1].minAndMaxValue[2]),Float32(numbb.value ))
        textureList[1].minAndMaxValue[2]= max(Float32(textureList[1].minAndMaxValue[2]),Float32(numbb.value ))

        
    end#if    


    updateImagesDisplayed(actor.actor.currentlyDispDat
    , actor.actor.mainForDisplayObjects
    , actor.actor.textDispObj
    , actor.actor.calcDimsStruct ,valueForMasToSett )
# for undoing action
if(toBeSavedForBack)
    addToforUndoVector(actor, ()-> processKeysInfo( Option(old),actor, keyInfo,false ))
end

end#processKeysInfo



"""
In order to enable undoing last action we just invoke last function from list 
"""
function processKeysInfoUndo(numbb::Identity{Bool},actor::SyncActor{Any, ActorWithOpenGlObjects},keyInfo::KeyboardStruct ) where T
    if(!isempty(actor.actor.forUndoVector))
     pop!(actor.actor.forUndoVector)()
    end
end#processKeysInfo



"""
when tab plus will be pressed it will increase stroke width
when tab minus will be pressed it will increase stroke width
"""
function processKeysInfo(annot::Identity{AnnotationStruct}
    ,actor::SyncActor{Any, ActorWithOpenGlObjects}
    ,keyInfo::KeyboardStruct 
    ,toBeSavedForBack::Bool = true) where T

    textureList = actor.actor.textureToModifyVec
    if (!isempty(textureList))
        texture= textureList[1]
        oldsWidth = texture.strokeWidth
        texture.strokeWidth= oldsWidth+=annot.value.strokeWidthChange
        # for undoing action
        if(toBeSavedForBack)
        addToforUndoVector(actor, ()-> processKeysInfo( Option(AnnotationStruct(oldsWidth)),actor, keyInfo,false ))
        end#if
    end#if

end#processKeysInfo

end#OtherKeyboardActions