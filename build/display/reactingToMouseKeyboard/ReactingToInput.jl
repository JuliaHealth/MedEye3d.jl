
module ReactingToInput
using Rocket, GLFW,ModernGL,Setfield,  ..ReactToScroll,  ..ForDisplayStructs
using  ..TextureManag,DataTypesBasic,  ..ReactOnMouseClickAndDrag,  ..ReactOnKeyboard,  ..DataStructs,  ..StructsManag,  ..DisplayWords
using ..MaskDiffrence, ..KeyboardVisibility, ..OtherKeyboardActions, ..WindowControll, ..ChangePlane
export subscribeGLFWtoActor


"""
adding the data into about openGL and GLFW context to enable proper display of main image and masks
"""
function setUpMainDisplay(mainForDisplayObjects:: forDisplayObjects,actor::SyncActor{Any, ActorWithOpenGlObjects})
    actor.actor.mainForDisplayObjects.stopListening[]=true

    actor.actor.mainForDisplayObjects=mainForDisplayObjects
    actor.actor.mainForDisplayObjects.stopListening[]=false

end#setUpMainDisplay

"""
adding the data needed for text display; also activates appropriate quad for the display
    it also configures texture that is build for text display
"""
function setUpWordsDisplay(textDispObject:: ForWordsDispStruct,actor::SyncActor{Any, ActorWithOpenGlObjects})

    actor.actor.mainForDisplayObjects.stopListening[]=true


     bindAndActivateForText(textDispObject.shader_program_words 
    , textDispObject.fragment_shader_words
    ,actor.actor.mainForDisplayObjects.vertex_shader
    ,textDispObject.vbo_words
    ,actor.actor.calcDimsStruct)

    texId =  createTexture(UInt8,actor.actor.calcDimsStruct.textTexturewidthh
                            ,actor.actor.calcDimsStruct.textTextureheightt
                            ,GL_R8UI,GL_UNSIGNED_BYTE)

    textSpec= setproperties(textDispObject.textureSpec,(ID=texId) )

    samplerRef= glGetUniformLocation(textDispObject.shader_program_words, "TextTexture1")
    
    glUniform1i(samplerRef,length(actor.actor.mainForDisplayObjects.listOfTextSpecifications)+1)
    textDispObjectiNITIALIZED= setproperties(textDispObject,(textureSpec=textSpec) )
    
    actor.actor.textDispObj=textDispObjectiNITIALIZED
    # now reactivating the main vbo and shader program


    reactivateMainObj(actor.actor.mainForDisplayObjects.shader_program
    , actor.actor.mainForDisplayObjects.vbo
    ,actor.actor.calcDimsStruct    )
    
    actor.actor.mainForDisplayObjects.stopListening[]=false

end#setUpWordsDisplay


"""
adding the data about 3 dimensional arrays that will be source of data used for scrolling behaviour
onScroll Data - list of tuples where first is the name of the texture that we provided and second is associated data (3 dimensional array of appropriate type)

"""
function setUpForScrollData(onScrollData::FullScrollableDat ,actor::SyncActor{Any, ActorWithOpenGlObjects})
    actor.actor.mainForDisplayObjects.stopListening[]=true
    
    onScrollData.slicesNumber= getSlicesNumber(onScrollData)
    actor.actor.onScrollData=onScrollData
    #In order to refresh all in case we would change the texture dimensions ...
    ChangePlane.processKeysInfo(Option(onScrollData.dataToScrollDims),actor,KeyboardStruct()  )
      #so  It will precalculate some data and later mouse modification will be swift
      oldd = actor.actor.valueForMasToSet 
      
      actor.actor.valueForMasToSet = valueForMasToSetStruct(value = 0)
      ReactOnMouseClickAndDrag.reactToMouseDrag(MouseStruct(true,false, [CartesianIndex(5,5)]),actor )
      actor.actor.valueForMasToSet = oldd

  actor.actor.mainForDisplayObjects.stopListening[]=false

end#setUpMainDisplay


"""
add data needed for proper calculations of mouse, verticies positions ... etc
"""
function setUpCalcDimsStruct(calcDim::CalcDimsStruct ,actor::SyncActor{Any, ActorWithOpenGlObjects})
    actor.actor.mainForDisplayObjects.stopListening[]=true

    actor.actor.calcDimsStruct =calcDim
    
    actor.actor.mainForDisplayObjects.stopListening[]=false

end#setUpCalcDimsStruct



"""
sets value we are setting to the  active mask vie mause interaction, in case mask is modifiable 
"""
function setUpvalueForMasToSet(valueForMasToSett::valueForMasToSetStruct ,actor::SyncActor{Any, ActorWithOpenGlObjects})
    actor.actor.mainForDisplayObjects.stopListening[]=true

    actor.actor.valueForMasToSet =valueForMasToSett
    
    updateImagesDisplayed(actor.actor.currentlyDispDat
    , actor.actor.mainForDisplayObjects
    , actor.actor.textDispObj
    , actor.actor.calcDimsStruct ,valueForMasToSett )

    actor.actor.mainForDisplayObjects.stopListening[]=false

end#setUpvalueForMasToSet


"""
enables updating just a single slice that is displayed - do not change what will happen after scrolling
one need to pass data to actor in 
struct that holds tuple where first entry is
-vector of tuples whee first entry in tuple is name of texture given in the setup and second is 2 dimensional aray of appropriate type with image data
- Int - second is Int64 - that is marking the screen number to which we wan to set the actor state
"""
function updateSingleImagesDisplayedSetUp(singleSliceDat::SingleSliceDat ,actor::SyncActor{Any, ActorWithOpenGlObjects})
    actor.actor.mainForDisplayObjects.stopListening[]=true
    updateImagesDisplayed(singleSliceDat
                        , actor.actor.mainForDisplayObjects
                        , actor.actor.textDispObj
                        , actor.actor.calcDimsStruct
                        ,actor.actor.valueForMasToSet  )
     
                        
    actor.actor.currentlyDispDat=singleSliceDat
    actor.actor.currentDisplayedSlice = singleSliceDat.sliceNumber
    actor.actor.isSliceChanged = true # mark for mouse interaction that we changed slice
   
    actor.actor.mainForDisplayObjects.stopListening[]=false

end #updateSingleImagesDisplayed



"""
configuring actor using multiple dispatch mechanism in order to connect input to proper functions; this is not 
encapsulated by a function becouse this is configuration of Rocket and needs to be global
"""

Rocket.on_next!(actor::SyncActor{Any, ActorWithOpenGlObjects}, data::Int64) = reactToScroll(data,actor )
Rocket.on_next!(actor::SyncActor{Any, ActorWithOpenGlObjects}, data:: forDisplayObjects) = setUpMainDisplay(data,actor)
Rocket.on_next!(actor::SyncActor{Any, ActorWithOpenGlObjects}, data:: ForWordsDispStruct) = setUpWordsDisplay(data,actor)
Rocket.on_next!(actor::SyncActor{Any, ActorWithOpenGlObjects}, data::CalcDimsStruct) = setUpCalcDimsStruct(data,actor)
Rocket.on_next!(actor::SyncActor{Any, ActorWithOpenGlObjects}, data::valueForMasToSetStruct) = setUpvalueForMasToSet(data,actor)


Rocket.on_next!(actor::SyncActor{Any, ActorWithOpenGlObjects}, data::FullScrollableDat) = setUpForScrollData(data,actor)
Rocket.on_next!(actor::SyncActor{Any, ActorWithOpenGlObjects}, data::SingleSliceDat) = updateSingleImagesDisplayedSetUp(data,actor)
Rocket.on_next!(actor::SyncActor{Any, ActorWithOpenGlObjects}, data::MouseStruct) = reactToMouseDrag(data,actor)
Rocket.on_next!(actor::SyncActor{Any, ActorWithOpenGlObjects}, data::KeyboardStruct) = reactToKeyboard(data,actor)

Rocket.on_error!(actor::SyncActor{Any, ActorWithOpenGlObjects}, err)      = error(err)
Rocket.on_complete!(actor::SyncActor{Any, ActorWithOpenGlObjects})        = ""




"""
when GLFW context is ready we need to use this  function in order to register GLFW events to Rocket actor - we use subscription for this
    actor - Roctet actor that holds objects needed for display like window etc...  
    return list of subscriptions so if we will need it we can unsubscribe
"""
function subscribeGLFWtoActor(actor ::SyncActor{Any, ActorWithOpenGlObjects})

    #controll scrolling
    forDisplayConstants = actor.actor.mainForDisplayObjects

    scrollback=  ReactToScroll.registerMouseScrollFunctions(forDisplayConstants.window,forDisplayConstants.stopListening,actor.actor.isBusy)
    GLFW.SetScrollCallback(forDisplayConstants.window, (a, xoff, yoff) -> scrollback(a, xoff, yoff))

    keyBoardAct = registerKeyboardFunctions(forDisplayConstants.window,forDisplayConstants.stopListening)
    buttonSubs  = registerMouseClickFunctions(forDisplayConstants.window,forDisplayConstants.stopListening,actor.actor.calcDimsStruct,actor.actor.isBusy )
  


    keyboardSub = subscribe!(keyBoardAct, actor)
    scrollSubscription = subscribe!(scrollback, actor)
    mouseClickSub = subscribe!(buttonSubs, actor)



return [scrollSubscription,mouseClickSub,keyboardSub]

end






end #ReactToGLFWInpuut