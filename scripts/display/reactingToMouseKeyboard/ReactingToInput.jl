using DrWatson
@quickactivate "Probabilistic medical segmentation"

module ReactingToInput
using Rocket, GLFW,ModernGL,Setfield, Main.ReactToScroll, Main.ForDisplayStructs
using Main.TextureManag, Main.ReactOnMouseClickAndDrag, Main.ReactOnKeyboard, Main.DataStructs, Main.StructsManag, Main.DisplayWords

export subscribeGLFWtoActor


```@doc
adding the data into about openGL and GLFW context to enable proper display
```
function setUpMainDisplay(mainForDisplayObjects::Main.ForDisplayStructs.forDisplayObjects,actor::SyncActor{Any, ActorWithOpenGlObjects})
    actor.actor.mainForDisplayObjects.stopListening[]=true

    actor.actor.mainForDisplayObjects=mainForDisplayObjects
    actor.actor.mainForDisplayObjects.stopListening[]=false

end#setUpMainDisplay

```@doc
adding the data needed for text display
    it also configures texture that is build for text display
```
function setUpWordsDisplay(textDispObject::Main.ForDisplayStructs.ForWordsDispStruct,actor::SyncActor{Any, ActorWithOpenGlObjects})

    actor.actor.mainForDisplayObjects.stopListening[]=true


     bindAndActivateForText(textDispObject.shader_program_words 
    , textDispObject.fragment_shader_words
    ,textDispObject.vbo_words
    ,actor.actor.calcDimsStruct)

    texId =  createTexture(0,textDispObject.textureSpec.widthh 
                            ,textDispObject.textureSpec.heightt
                            ,GL_R8UI)

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


setUpForScrollDataStr= """
adding the data about 3 dimensional arrays that will be source of data used for scrolling behaviour
onScroll Data - list of tuples where first is the name of the texture that we provided and second is associated data (3 dimensional array of appropriate type)

"""
@doc setUpForScrollDataStr
function setUpForScrollData(onScrollData::FullScrollableDat ,actor::SyncActor{Any, ActorWithOpenGlObjects})
    actor.actor.mainForDisplayObjects.stopListening[]=true

    actor.actor.onScrollData=onScrollData
    actor.actor.mainForDisplayObjects.slicesNumber= getSlicesNumber(onScrollData)
    
    actor.actor.mainForDisplayObjects.stopListening[]=false

end#setUpMainDisplay


setUpCalcDimsStructStr= """
add data needed for proper calculations of mouse, verticies positions ... etc
"""
@doc setUpCalcDimsStructStr
function setUpCalcDimsStruct(calcDim::CalcDimsStruct ,actor::SyncActor{Any, ActorWithOpenGlObjects})
    actor.actor.mainForDisplayObjects.stopListening[]=true

    actor.actor.calcDimsStruct =calcDim
    
    actor.actor.mainForDisplayObjects.stopListening[]=false

end#setUpCalcDimsStruct



updateSingleImagesDisplayedSetUpStr =    """
enables updating just a single slice that is displayed - do not change what will happen after scrolling
one need to pass data to actor in 
struct that holds tuple where first entry is
-vector of tuples whee first entry in tuple is name of texture given in the setup and second is 2 dimensional aray of appropriate type with image data
- Int - second is Int64 - that is marking the screen number to which we wan to set the actor state
"""
@doc updateSingleImagesDisplayedSetUpStr
function updateSingleImagesDisplayedSetUp(singleSliceDat::SingleSliceDat ,actor::SyncActor{Any, ActorWithOpenGlObjects})
    actor.actor.mainForDisplayObjects.stopListening[]=true
    updateImagesDisplayed(singleSliceDat, actor.actor.mainForDisplayObjects)
    actor.actor.currentlyDispDat=singleSliceDat
    actor.actor.currentDisplayedSlice = singleSliceDat.sliceNumber
    actor.actor.isSliceChanged = true # mark for mouse interaction that we changed slice
   
    actor.actor.mainForDisplayObjects.stopListening[]=false

end #updateSingleImagesDisplayed



"""
configuring actor using multiple dispatch mechanism in order to connect input to proper functions; this is not 
encapsulated by a function becouse this is configuration of Rocket and needs to be global
"""

Rocket.on_next!(actor::SyncActor{Any, ActorWithOpenGlObjects}, data::Bool) = reactToScroll(data,actor )
Rocket.on_next!(actor::SyncActor{Any, ActorWithOpenGlObjects}, data::Main.ForDisplayStructs.forDisplayObjects) = setUpMainDisplay(data,actor)
Rocket.on_next!(actor::SyncActor{Any, ActorWithOpenGlObjects}, data::Main.ForDisplayStructs.ForWordsDispStruct) = setUpWordsDisplay(data,actor)
Rocket.on_next!(actor::SyncActor{Any, ActorWithOpenGlObjects}, data::CalcDimsStruct) = setUpCalcDimsStruct(data,actor)


Rocket.on_next!(actor::SyncActor{Any, ActorWithOpenGlObjects}, data::FullScrollableDat) = setUpForScrollData(data,actor)
Rocket.on_next!(actor::SyncActor{Any, ActorWithOpenGlObjects}, data::SingleSliceDat) = updateSingleImagesDisplayedSetUp(data,actor)
Rocket.on_next!(actor::SyncActor{Any, ActorWithOpenGlObjects}, data::Vector{CartesianIndex{2}}) = reactToMouseDrag(data,actor)
Rocket.on_next!(actor::SyncActor{Any, ActorWithOpenGlObjects}, data::KeyboardStruct) = reactToKeyboard(data,actor)

Rocket.on_error!(actor::SyncActor{Any, ActorWithOpenGlObjects}, err)      = error(err)
Rocket.on_complete!(actor::SyncActor{Any, ActorWithOpenGlObjects})        = println("Completed!")




```@doc
when GLFW context is ready we need to use this  function in order to register GLFW events to Rocket actor - we use subscription for this
    actor - Roctet actor that holds objects needed for display like window etc...  
    return list of subscriptions so if we will need it we can unsubscribe
```
function subscribeGLFWtoActor(actor ::SyncActor{Any, ActorWithOpenGlObjects})

    #controll scrolling
    forDisplayConstants = actor.actor.mainForDisplayObjects

    scrollback= Main.ReactToScroll.registerMouseScrollFunctions(forDisplayConstants.window,forDisplayConstants.stopListening)
    GLFW.SetScrollCallback(forDisplayConstants.window, (a, xoff, yoff) -> scrollback(a, xoff, yoff))

    keyBoardAct = registerKeyboardFunctions(forDisplayConstants.window,forDisplayConstants.stopListening)
    buttonSubs = registerMouseClickFunctions(forDisplayConstants.window,forDisplayConstants.stopListening,actor.actor.calcDimsStruct )
  
    keyboardSub = subscribe!(keyBoardAct, actor)
    scrollSubscription = subscribe!(scrollback, actor)
    mouseClickSub = subscribe!(buttonSubs, actor)

return [scrollSubscription,mouseClickSub,keyboardSub]

end






end #ReactToGLFWInpuut