"""
controls changing plane for example from transverse to saggital ...
"""
module ChangePlane
using ModernGL, ..DisplayWords, ..StructsManag, Setfield, ..PrepareWindow,   ..DataStructs ,Glutils, Rocket, GLFW,Dictionaries,  ..ForDisplayStructs, ..TextureManag,  ..OpenGLDisplayUtils,  ..Uniforms, Match, Parameters,DataTypesBasic   

"""
In case we want to change the dimansion of scrolling so for example from transverse 
    toBeSavedForBack - just marks weather we wat to save the info how to undo latest action
    - false if we invoke it from undoing 
"""

function processKeysInfo(toScrollDatPrim::Identity{DataToScrollDims}
                    ,actor::SyncActor{Any, ActorWithOpenGlObjects}
                    ,keyInfo::KeyboardStruct
                    ,toBeSavedForBack::Bool = true ) where T
    toScrollDat= toScrollDatPrim.value

    old = actor.actor.onScrollData.dimensionToScroll

    newCalcDim= getHeightToWidthRatio(actor.actor.calcDimsStruct,toScrollDat )|>
                    getMainVerticies
     actor.actor.calcDimsStruct = newCalcDim
#In order to make the  background black  before we will render quad of possibly diffrent dimensions we will set all to invisible - and obtain black background
textSpecs = actor.actor.mainForDisplayObjects.listOfTextSpecifications

for textSpec in textSpecs
    setTextureVisibility(false,textSpec.uniforms )
end#for    
basicRender(actor.actor.mainForDisplayObjects.window)


    #we need to change textures only if dimensions do not match
  #  if(actor.actor.calcDimsStruct.imageTextureWidth!=newCalcDim.imageTextureWidth  || actor.actor.calcDimsStruct.imageTextureHeight!=newCalcDim.imageTextureHeight )
        # first we need to update information about dimensions etc 


        #next we need to delete all textures and create new ones 

        arr = map(it->it.ID[],textSpecs)
glFinish()# make open gl ready for work

glDeleteTextures(length(arr), arr)# deleting

        #getting new 
        initializeTextures(textSpecs,newCalcDim)

   # end#if


actor.actor.onScrollData.dimensionToScroll = toScrollDat.dimensionToScroll

actor.actor.onScrollData.slicesNumber = getSlicesNumber(actor.actor.onScrollData)
#getting  the slice of intrest based on last recorded mouse position 

current=actor.actor.lastRecordedMousePosition[toScrollDat.dimensionToScroll]

#displaying all


singleSlDat= actor.actor.onScrollData.dataToScroll|>
(scrDat)-> map(threeDimDat->threeToTwoDimm(threeDimDat.type,Int64(current),toScrollDat.dimensionToScroll,threeDimDat ),scrDat) |>
(twoDimList)-> SingleSliceDat(listOfDataAndImageNames=twoDimList
                            ,sliceNumber=current
                            ,textToDisp = getTextForCurrentSlice(actor.actor.onScrollData, Int32(current))  )

                            # glFinish()
                            # glFlush()
                            # glClearColor(0.0, 0.0, 0.0 , 1.0)
                            # GLFW.SwapBuffers(actor.actor.mainForDisplayObjects.window)

dispObj = actor.actor.mainForDisplayObjects
#for displaying new quad - to accomodate new proportions
reactivateMainObj(dispObj.shader_program, dispObj.vbo,newCalcDim  )

glClear(GL_COLOR_BUFFER_BIT)
actor.actor.currentlyDispDat=singleSlDat = singleSlDat
updateImagesDisplayed(singleSlDat
                    ,actor.actor.mainForDisplayObjects
                    ,actor.actor.textDispObj
                    ,newCalcDim 
                    ,actor.actor.valueForMasToSet      )





 #@info "singleSlDat" singleSlDat
     #saving information about current slice for future reference
actor.actor.currentDisplayedSlice = current
# to enbling getting back
if(toBeSavedForBack)
    addToforUndoVector(actor, ()-> processKeysInfo( Option(old),actor, keyInfo,false ))
end#if

end#processKeysInfo
end#ChangePlane