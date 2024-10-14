"""
controls changing plane for example from transverse to saggital ...
"""
module ChangePlane
using ModernGL, GLFW, Dictionaries, Parameters, DataTypesBasic, Setfield
using ..DisplayWords, ..StructsManag, ..PrepareWindow, ..DataStructs, ..ForDisplayStructs, ..TextureManag, ..OpenGLDisplayUtils, ..Uniforms

"""
In case we want to change the dimansion of scrolling so for example from transverse to coronal ...
    toBeSavedForBack - just marks weather we wat to save the info how to undo latest action
    - false if we invoke it from undoing
"""

function processKeysInfo(toScrollDatPrim::Identity{DataToScrollDims}, stateObject::StateDataFields, keyInfo::KeyboardStruct, toBeSavedForBack::Bool=true)


    toScrollDat = toScrollDatPrim.value

    old = stateObject.onScrollData.dimensionToScroll

    ratioSetcalcDim = getHeightToWidthRatio(stateObject.calcDimsStruct, toScrollDat)
    newCalcDim = getMainVerticies(ratioSetcalcDim, stateObject.displayMode, stateObject.imagePosition)

    stateObject.calcDimsStruct = newCalcDim
    #In order to make the  background black  before we will render quad of possibly diffrent dimensions we will set all to invisible - and obtain black background
    textSpecs = stateObject.mainForDisplayObjects.listOfTextSpecifications

    for textSpec in textSpecs
        setTextureVisibility(false, textSpec.uniforms)
    end#for
    basicRender(stateObject.mainForDisplayObjects.window)


    #we need to change textures only if dimensions do not match
    #  if(actor.actor.calcDimsStruct.imageTextureWidth!=newCalcDim.imageTextureWidth  || actor.actor.calcDimsStruct.imageTextureHeight!=newCalcDim.imageTextureHeight )
    # first we need to update information about dimensions etc


    #next we need to delete all textures and create new ones

    arr = map(it -> it.ID[], textSpecs)
    glFinish()# make open gl ready for work

    glDeleteTextures(length(arr), arr)# deleting

    #getting new

    initializeTextures(textSpecs, newCalcDim)

    # end#if


    stateObject.onScrollData.dimensionToScroll = toScrollDat.dimensionToScroll
    stateObject.onScrollData.dataToScrollDims = toScrollDat
    stateObject.onScrollData.slicesNumber = getSlicesNumber(stateObject.onScrollData)
    #getting  the slice of intrest based on last recorded mouse position

    current = stateObject.lastRecordedMousePosition[toScrollDat.dimensionToScroll]

    #displaying all


    singleSlDat = stateObject.onScrollData.dataToScroll |>
                  (scrDat) -> map(threeDimDat -> threeToTwoDimm(threeDimDat.type, Int64(current), toScrollDat.dimensionToScroll, threeDimDat), scrDat) |>
                              (twoDimList) -> SingleSliceDat(listOfDataAndImageNames=twoDimList, sliceNumber=current, textToDisp=getTextForCurrentSlice(stateObject.onScrollData, Int32(current)))

    # glFinish()
    # glFlush()
    # glClearColor(0.0, 0.0, 0.0 , 1.0)
    # GLFW.SwapBuffers(actor.actor.mainForDisplayObjects.window)

    dispObj = stateObject.mainForDisplayObjects
    #for displaying new quad - to accomodate new proportions

    reactivateMainObj(dispObj.shader_program, dispObj.vbo, newCalcDim)



    glClear(GL_COLOR_BUFFER_BIT)


    stateObject.currentlyDispDat = singleSlDat = singleSlDat

    updateImagesDisplayed(singleSlDat, stateObject.mainForDisplayObjects, stateObject.textDispObj, newCalcDim, stateObject.valueForMasToSet, stateObject.crosshairFields, stateObject.mainRectFields)



    #saving information about current slice for future reference
    stateObject.currentDisplayedSlice = current
    # to enbling getting back
    # if(toBeSavedForBack)
    #     addToforUndoVector(stateObject, ()-> processKeysInfo( Option(old),stateObject, keyInfo,false ))
    # end


end#processKeysInfo
end#ChangePlane


