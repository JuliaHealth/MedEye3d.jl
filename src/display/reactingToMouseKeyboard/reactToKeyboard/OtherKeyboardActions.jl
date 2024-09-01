"""
functions to controll stroke width , setting which texture is currently active and actions undoing
"""
module OtherKeyboardActions
using ModernGL, Setfield, GLFW, Dictionaries, Parameters, DataTypesBasic
using ..DisplayWords, ..StructsManag, ..PrepareWindow, ..DataStructs, ..ForDisplayStructs, ..TextureManag, ..OpenGLDisplayUtils, ..Uniforms

"""
in case we want to  get new number set for manual modifications
    toBeSavedForBack - just marks weather we wat to save the info how to undo latest action
    - false if we invoke it from undoing
"""
function processKeysInfo(numbb::Identity{Int64}, stateObject::StateDataFields, keyInfo::KeyboardStruct, toBeSavedForBack::Bool=true)

    # @info "nnnnn $(numbb.value)"
    valueForMasToSett = valueForMasToSetStruct(value=numbb.value)
    old = stateObject.valueForMasToSet.value
    stateObject.valueForMasToSet = valueForMasToSett
    textureList = stateObject.textureToModifyVec

    # in case we increase number it should not be outside of the possible values
    if (!isempty(textureList))
        # @info max(Float32(textureList[1].minAndMaxValue[2]), Float32(numbb.value))
        textureList[1].minAndMaxValue[2] = max(Float32(textureList[1].minAndMaxValue[2]), Float32(numbb.value))


    end#if


    updateImagesDisplayed(stateObject.currentlyDispDat, stateObject.mainForDisplayObjects, stateObject.textDispObj, stateObject.calcDimsStruct, valueForMasToSett)
    # for undoing action
    # if(toBeSavedForBack)
    #     addToforUndoVector(stateObject, ()-> processKeysInfo( Option(old),stateObject, keyInfo,false ))
    # end

end#processKeysInfo



"""
In order to enable undoing last action we just invoke last function from list
"""
function processKeysInfoUndo(numbb::Identity{Bool}, stateObject::StateDataFields, keyInfo::KeyboardStruct)
    if (!isempty(stateObject.forUndoVector))
        pop!(stateObject.forUndoVector)()
    end
end#processKeysInfo



"""
when tab plus will be pressed it will increase stroke width
when tab minus will be pressed it will increase stroke width
"""
function processKeysInfo(annot::Identity{AnnotationStruct}, stateObject::StateDataFields, keyInfo::KeyboardStruct, toBeSavedForBack::Bool=true)

    textureList = stateObject.textureToModifyVec
    if (!isempty(textureList))
        texture = textureList[1]
        oldsWidth = texture.strokeWidth
        texture.strokeWidth = oldsWidth += annot.value.strokeWidthChange
        # for undoing action
        # if (toBeSavedForBack)
        #     addToforUndoVector(stateObject, () -> processKeysInfo(Option(AnnotationStruct(oldsWidth)), stateObject, keyInfo, false))
        # end#if
    end#if

end#processKeysInfo

end#OtherKeyboardActions
