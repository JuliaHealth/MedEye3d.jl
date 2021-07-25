using DrWatson
@quickactivate "Probabilistic medical segmentation"

module ReactingToInput
using Rocket
using GLFW
export registerMouseScrollFunctions

ScrollCallbackSubscribableStr="""
uploading data to given texture; of given types associated
returns subscription in order to enable unsubscribing in the end 
"""
@doc ScrollCallbackSubscribableStr
mutable struct ScrollCallbackSubscribable <: Subscribable{Int}
    xoff_previous :: Float64
    yoff_previous :: Float64
    subject :: Subject{Int}

    ScrollCallbackSubscribable() = new(0.0, 0.0, Subject(Int))
end
onSubscribeStr="""
configuting Rocket on Subscribe
"""
@doc onSubscribeStr
function Rocket.on_subscribe!(handler::ScrollCallbackSubscribable, actor)
    return subscribe!(handler.subject, actor)
end

handlerStr="""
place where we can put the logic  
"""
@doc handlerStr
function (handler::ScrollCallbackSubscribable)(_, xoff, yoff)
      if handler.yoff_previous > yoff
          next!(handler.subject, 1)
      else
          next!(handler.subject, -1)
      end 
      handler.xoff_previous = xoff
      handler.yoff_previous = yoff
end


registerMouseScrollFunctionsStr="""
uploading data to given texture; of given types associated
returns subscription in order to enable unsubscribing in the end 
window - GLFW window 
return subscription - we can when closing then  unsubscribe - to tidy up all
"""
@doc registerMouseScrollFunctionsStr
function registerMouseScrollFunctions(window)

scrollback = ScrollCallbackSubscribable()
stopListening[]=true # stoping event listening loop to free the GLFW context

GLFW.SetScrollCallback(window, (a, xoff, yoff) -> scrollback(a, xoff, yoff))

subscription = subscribe!(scrollback, (direction) -> println(direction))


stopListening[]=false # reactivate event listening loop

end #registerMouseScrollFunctions




```@doc
Actor that is able to store a state to keep needed data for proper display
```
mutable struct ActorWithOpenGlObjects <: NextActor{Any}
    currentDisplayedSlice::Int # stores information what slice number we are currently displaying
    mainForDisplayObjects::Main.ForDisplayStructs.forDisplayObjects # stores objects needed to  display using OpenGL and GLFW
    ActorWithOpenGlObjects() = new(1,forDisplayObjects())
end


```@doc
in case of the scroll p true will be send in case of down - false
in response to it it sets new screen int variable and changes displayed screen
```
function reactToScroll(isScrollUp::Bool, actor::ActorWithOpenGlObjects)
    current = actor.currentDisplayedSlice
    isScrollUp ? current+=1 : current-=1
   # we do not want to move outside of possible range of slices
   lastSlice = actor.mainForDisplayObjects.listOfTextSpecifications[1].slicesNumber
    if(current<0) current=0 end 
    if(current>=lastSlice) current=lastSlice end 
    #logic to change displayed screen

    listOfDataAndImageNames = [("grandTruthLiverLabel",exampleLabels[current,:,:]),("mainCTImage",exampleDat[current,:,:] )]
    Main.SegmentationDisplay.updateImagesDisplayed(listOfDataAndImageNames
         ,actor.mainForDisplayObjects )
    
         #saving information about current slice for future reference
    actor.currentDisplayedSlice = current

end#reactToScroll
```@doc
adding the data into actor to enable proper display
```
function setUpMainDisplay(mainForDisplayObjects::Main.ForDisplayStructs.forDisplayObjects,actor::ActorWithOpenGlObjects)
    actor.mainForDisplayObjects=mainForDisplayObjects

end#setUpMainDisplay



Rocket.on_next!(actor::ActorWithOpenGlObjects, data::Bool) = reactToScroll(data,actor )
Rocket.on_next!(actor::ActorWithOpenGlObjects, data::Main.ForDisplayStructs.forDisplayObjects) = setUpMainDisplay(data,actor)
Rocket.on_error!(actor::ActorWithOpenGlObjects, err)      = error(err)
Rocket.on_complete!(actor::ActorWithOpenGlObjects)        = println("Completed!")






end #ReactToGLFWInpuut