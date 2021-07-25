using DrWatson
@quickactivate "Probabilistic medical segmentation"


ReactOnMouseClickAndDragSTR="""
module 
code adapted from https://discourse.julialang.org/t/custom-subject-in-rocket-jl-for-mouse-events-from-glfw/65133/3
"""
#@doc ReactOnMouseClickAndDragSTR
module ReactOnMouseClickAndDrag
using Rocket
using GLFW
using Main.ForDisplayStructs
using Main.TextureManag
using Logging



MouseCallbackSubscribableStr="""
struct that enables reacting to  the input from scrolling
"""
@doc MouseCallbackSubscribableStr
mutable struct MouseCallbackSubscribable <: Subscribable{(CartesianIndex,Bool)}
    subject :: Subject{Bool}
    ScrollCallbackSubscribable() = new(Subject(Bool, scheduler = AsyncScheduler())) # if value is true it means we scroll up if false we scroll down
end



```@doc
configuting Rocket on Subscribe so we get custom handler of input as we see we still need to define actor
```
function Rocket.on_subscribe!(handler::ScrollCallbackSubscribable, actor)
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
function registerMouseScrollFunctions(window,stopListening)

scrollback = ScrollCallbackSubscribable()
stopListening[]=true # stoping event listening loop to free the GLFW context

GLFW.SetScrollCallback(window, (a, xoff, yoff) -> scrollback(a, xoff, yoff))

stopListening[]=false # reactivate event listening loop

return scrollback

end #registerMouseScrollFunctions





end #ReactOnMouseClickAndDrag
