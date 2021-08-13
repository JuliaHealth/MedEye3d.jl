using DrWatson
@quickactivate "Probabilistic medical segmentation"


ReactToScrollStr="""
module that holds functions needed to  react to scrolling
Generally first we need to pass the GLFW callback to the Rocket obeservable 
code adapted from https://discourse.julialang.org/t/custom-subject-in-rocket-jl-for-mouse-events-from-glfw/65133/3
"""
#@doc ReactToScrollStr
module ReactToScroll
using Rocket, GLFW, Main.ForDisplayStructs, Main.TextureManag,Logging, Main.DataStructs, Main.StructsManag

export reactToScroll
export registerMouseScrollFunctions

ScrollCallbackSubscribableStr="""
struct that enables reacting to  the input from scrolling
"""
@doc ScrollCallbackSubscribableStr
mutable struct ScrollCallbackSubscribable <: Subscribable{Bool}
    subject :: Subject{Bool}
    ScrollCallbackSubscribable() = new(Subject(Bool, scheduler = AsyncScheduler())) # if value is true it means we scroll up if false we scroll down
end



```@doc
configuting Rocket on Subscribe so we get custom handler of input as we see we still need to define actor
```
function Rocket.on_subscribe!(handler::ScrollCallbackSubscribable, actor::SyncActor{Any, ActorWithOpenGlObjects})
    return subscribe!(handler.subject, actor)
end



handlerStr="""
we define how handler should act on the subject - observable so it will pass event onto subject
"""
@doc handlerStr
function (handler::ScrollCallbackSubscribable)(_, xoff, yoff)
          next!(handler.subject, yoff==1.0)#true if we scroll up
end




registerMouseScrollFunctionsStr="""
uploading data to given texture; of given types associated
returns subscription in order to enable unsubscribing in the end 
window - GLFW window 
stopListening - atomic boolean able to stop the event listening cycle
return scrollback - that holds boolean subject (observable) to which we can react by subscribing appropriate actor
"""
@doc registerMouseScrollFunctionsStr
function registerMouseScrollFunctions(window::GLFW.Window,stopListening::Base.Threads.Atomic{Bool})

scrollback = ScrollCallbackSubscribable()
stopListening[]=true # stoping event listening loop to free the GLFW context

GLFW.SetScrollCallback(window, (a, xoff, yoff) -> scrollback(a, xoff, yoff))

stopListening[]=false # reactivate event listening loop

return scrollback

end #registerMouseScrollFunctions



reactToScrollStr = """
in case of the scroll p true will be send in case of down - false
in response to it it sets new screen int variable and changes displayed screen
"""
@doc reactToScrollStr
function reactToScroll(isScrollUp::Bool, actor::SyncActor{Any, ActorWithOpenGlObjects})
    actor.actor.mainForDisplayObjects.stopListening[]=true
    current = actor.actor.currentDisplayedSlice
    isScrollUp ? current+=1 : current-=1

   # we do not want to move outside of possible range of slices
   lastSlice = actor.actor.mainForDisplayObjects.slicesNumber
   actor.actor.isSliceChanged = true
    if(current<1) current=1 end 
    if(lastSlice<1) lastSlice=1 end 
    if(current>=lastSlice) current=lastSlice end 

    #logic to change displayed screen
    #we select slice that we are intrested in
    singleSlDat= actor.actor.onScrollData.dataToScroll|>
    (scrDat)-> map(threeDimDat->threeToTwoDimm(threeDimDat.type,Int64(current),actor.actor.onScrollData.dimensionToScroll,threeDimDat ),scrDat) |>
    (twoDimList)-> SingleSliceDat(listOfDataAndImageNames=twoDimList,sliceNumber=current )
    
     updateImagesDisplayed(singleSlDat,actor.actor.mainForDisplayObjects )

     actor.actor.currentlyDispDat=singleSlDat

         #saving information about current slice for future reference
    actor.actor.currentDisplayedSlice = current

   actor.actor.mainForDisplayObjects.stopListening[]=false


end#reactToScroll




end #ReactToScroll