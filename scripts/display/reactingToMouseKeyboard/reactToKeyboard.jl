using DrWatson
@quickactivate "Probabilistic medical segmentation"



ReactOnKeyboardSTR="""
code adapted from https://discourse.julialang.org/t/custom-subject-in-rocket-jl-for-mouse-events-from-glfw/65133/3
module coordinating response to the  keyboard input - mainly shortcuts that  helps controlling  active/visible texture2D
"""
#@doc ReactOnKeyboardSTR
module ReactOnKeyboard
using Rocket
using GLFW
using Main.ForDisplayStructs
using Main.TextureManag
using Main.OpenGLDisplayUtils

KeyboardCallbackSubscribableStr= """
Object that enables managing input from keyboard - it stores the information also about
needed keys wheather they are kept pressed  
examples of keyboard input 
    action RELEASE GLFW.Action
    key s StringPRESS
    key s String
    action PRESS GLFW.Action
    key s StringRELEASE
    key s String
    action RELEASE GLFW.Action

"""
@doc KeyboardCallbackSubscribableStr
mutable struct KeyboardCallbackSubscribable <: Subscribable{CartesianIndex{2}}
# true when pressed and kept true until released
    isCtrlPressed::Bool
    isShiftPressed::Bool
    isAltPressed::Bool

    lasKeyPressed

    subject :: Subject{CartesianIndex{2}} 
   
end 

```@doc


```
function Rocket.on_subscribe!(handler::KeyboardCallbackSubscribable, actor::SyncActor{Any, ActorWithOpenGlObjects})
    return subscribe!(handler.subject, actor)
end


handlerStr="""

"""
@doc handlerStr
function (handler::KeyboardCallbackSubscribable)( a, x::Float64, y::Float64)
   # next!(handler.subject, CartesianIndex(Int(x),Int(y)))
 
end #handler

@doc handlerStr
function (handler::KeyboardCallbackSubscribable)(a, button::GLFW.MouseButton, action::GLFW.Action,m)
end #second handler



registerKeyboardFunctionsStr="""

"""
@doc registerKeyboardFunctionsStr
function registerKeyboardFunctions(window::GLFW.Window
                                    ,stopListening::Base.Threads.Atomic{Bool}
                                    )

return keyboardSubs

end #registerKeyboardFunctions


reactToKeyboardStr = """

"""
@doc reactToKeyboardStr
function reactToKeyboard(mouseCoord::CartesianIndex{2}, actor::SyncActor{Any, ActorWithOpenGlObjects})
    
    obj = actor.actor.mainForDisplayObjects
    
    obj.stopListening[]=true #free GLFW context
    
    
    
    obj.stopListening[]=false # reactivete event listening loop

    #send data for persistent storage TODO() modify for scrolling data 

end#reactToKeyboard






end #ReactOnMouseClickAndDrag
