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
mutable struct KeyboardCallbackSubscribable <: Subscribable{KeyboardStruct}
# true when pressed and kept true until released
# true if corresponding keys are kept pressed and become flase when relesed
    isCtrlPressed::Bool # left - scancode 37 right 105 - Int32
    isShiftPressed::Bool  # left - scancode 50 right 62- Int32
    isAltPressed::Bool# left - scancode 64 right 108- Int32
    lastKeyPressed::String # last pressed key 
    subject :: Subject{KeyboardStruct} 
   
end 

```@doc


```
function Rocket.on_subscribe!(handler::KeyboardCallbackSubscribable, actor::SyncActor{Any, ActorWithOpenGlObjects})
    return subscribe!(handler.subject, actor)
end


handlerStr="""

"""
@doc handlerStr
function (handler::KeyboardCallbackSubscribable)(str:String)
    handler.lastKeyPressed = str
   next!(handler.subject, KeyboardStruct(handler.isCtrlPressed, handler.isShiftPressed,handler., handler.   )  )
 
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
function reactToKeyboard(keybInfo::KeyboardStruct, actor::SyncActor{Any, ActorWithOpenGlObjects})
    
    obj = actor.actor.mainForDisplayObjects
    
    obj.stopListening[]=true #free GLFW context
    
    
    
    obj.stopListening[]=false # reactivete event listening loop

    #send data for persistent storage TODO() modify for scrolling data 

end#reactToKeyboard






end #ReactOnMouseClickAndDrag
