using DrWatson
@quickactivate "Probabilistic medical segmentation"

module ReactingToInput
using Rocket
using GLFW
using Main.ReactToScroll
using Main.ForDisplayStructs
using Main.TextureManag
using Main.ReactOnMouseClickAndDrag

export subscribeGLFWtoActor


```@doc
adding the data into about openGL and GLFW context to enable proper display
```
function setUpMainDisplay(mainForDisplayObjects::Main.ForDisplayStructs.forDisplayObjects,actor::SyncActor{Any, ActorWithOpenGlObjects})
    actor.actor.mainForDisplayObjects=mainForDisplayObjects

end#setUpMainDisplay



setUpForScrollDataStr= """
adding the data about 3 dimensional arrays that will be source of data used for scrolling behaviour
onScroll Data - list of tuples where first is the name of the texture that we provided and second is associated data (3 dimensional array of appropriate type)

"""
@doc setUpForScrollDataStr
function setUpForScrollData(onScrollData::Vector{Tuple{String, Array{T, 3} where T}} ,actor::SyncActor{Any, ActorWithOpenGlObjects})
    actor.actor.onScrollData=onScrollData
    @info "setting number of slices " size(onScrollData[1][2])[1]

    actor.actor.mainForDisplayObjects.slicesNumber= size(onScrollData[1][2])[1]

end#setUpMainDisplay



updateSingleImagesDisplayedSetUpStr =    """
enables updating just a single slice that is displayed - do not change what will happen after scrolling
one need to pass data to actor in 
tuple where first entry is
-vector of tuples whee first entry in tuple is name of texture given in the setup and second is 2 dimensional aray of appropriate type with image data
- Int - second is Int64 - that is marking the screen number to which we wan to set the actor state
"""
@doc updateSingleImagesDisplayedSetUpStr
function updateSingleImagesDisplayedSetUp(listOfDataAndImageNamesTuple::Tuple{Vector{Tuple{String, Array{T, 2} where T}},Int64} ,actor::SyncActor{Any, ActorWithOpenGlObjects})

updateImagesDisplayed(listOfDataAndImageNamesTuple[1], actor.actor.mainForDisplayObjects)
actor.actor.currentDisplayedSlice = listOfDataAndImageNamesTuple[2]

end #updateSingleImagesDisplayed



"""
configuring actor using multiple dispatch mechanism in order to connect input to proper functions; this is not 
encapsulated by a function becouse this is configuration of Rocket and needs to be global
"""

Rocket.on_next!(actor::SyncActor{Any, ActorWithOpenGlObjects}, data::Bool) = reactToScroll(data,actor )
Rocket.on_next!(actor::SyncActor{Any, ActorWithOpenGlObjects}, data::Main.ForDisplayStructs.forDisplayObjects) = setUpMainDisplay(data,actor)
Rocket.on_next!(actor::SyncActor{Any, ActorWithOpenGlObjects}, data::Vector{Tuple{String, Array{T, 3} where T}}) = setUpForScrollData(data,actor)
Rocket.on_next!(actor::SyncActor{Any, ActorWithOpenGlObjects}, data::Tuple{Vector{Tuple{String, Array{T, 2} where T}},Int64} ) = updateSingleImagesDisplayedSetUp(data,actor)

Rocket.on_next!(actor::SyncActor{Any, ActorWithOpenGlObjects}, data::CartesianIndex{2}) = reactToMouseDrag(data,actor)

Rocket.on_error!(actor::SyncActor{Any, ActorWithOpenGlObjects}, err)      = error(err)
Rocket.on_complete!(actor::SyncActor{Any, ActorWithOpenGlObjects})        = println("Completed!")


```@doc
In order to enable keyboard shortcuts 
```




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
    buttonSubs = registerMouseClickFunctions(forDisplayConstants.window,forDisplayConstants.stopListening)
    scrollSubscription = subscribe!(scrollback, actor)
    mouseClickSub = subscribe!(buttonSubs, actor)

return [scrollSubscription, mouseClickSub]

end






end #ReactToGLFWInpuut