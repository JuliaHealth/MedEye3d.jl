using DrWatson
@quickactivate "Probabilistic medical segmentation"


ReactOnMouseClickAndDragSTR="""
module 
code adapted from https://discourse.julialang.org/t/custom-subject-in-rocket-jl-for-mouse-events-from-glfw/65133/3
it is design to help processing data from 
    -GLFW.SetCursorPosCallback(window, (_, x, y) -> println("cursor: x, y")) and  for example : cursor: 29.0, 469.0  types   Float64  Float64   
    -GLFW.SetMouseButtonCallback(window, (_, button, action, mods) -> println("button action"))  for example types MOUSE_BUTTON_1 PRESS   GLFW.MouseButton  GLFW.Action 
The main function is to mark the interaction of the mouse to be saved in appropriate mask and be rendered onto the screen
so we modify the data that is the basis of the mouse interaction mask  and we pass the data on so appropriate part of the texture would be modified to be displayed on screen

"""
#@doc ReactOnMouseClickAndDragSTR
module ReactOnMouseClickAndDrag
using Rocket
using GLFW
using Main.ForDisplayStructs
using Main.TextureManag
using Logging


MouseCallbackSubscribableStr="""
struct that enables reacting to  the input  from mouse click  and drag the input will be 
    Cartesian index represening (x,y) x and y position  of the mouse - will be recorded only if left mouse button is pressed or keep presssed
"""
@doc MouseCallbackSubscribableStr
mutable struct MouseCallbackSubscribable <: Subscribable{CartesianIndex{2}}
    #true if left button is presed down - we make it true if the left button is pressed over image and false if mouse get out of the window or we get information about button release
    isLeftButtonDown ::Bool 
    #coordinates marking 4 corners of 
    #the quad that displays our medical image with the masks
    xmin::Int32
    ymin::Int32
    xmax::Int32
    ymax::Int32
    subject :: Subject{CartesianIndex{2}} # coordinates of mouse 
   
end



```@doc
configuting Rocket on Subscribe so we get custom handler of input as we see we still need to define actor
```
# function Rocket.on_subscribe!(handler::MouseCallbackSubscribable, actor::SyncActor{Any, ActorWithOpenGlObjects})
#     return subscribe!(handler.subject, actor)
# end

function Rocket.on_subscribe!(handler::MouseCallbackSubscribable, actor::SyncActor{Any, ActorWithOpenGlObjects})
    return subscribe!(handler.subject, actor)
end


handlerStr="""
we define how handler should act on the subject - observable so it will pass event onto subject - here we have 2 events that we want to be ready for - mouse button press
example of possible inputs that we would be intrested in 
for example : cursor: 29.0, 469.0  types   Float64  Float64   
for example  MOUSE_BUTTON_1 PRESS    types GLFW.MouseButton  GLFW.Action 
             MOUSE_BUTTON_1 RELEASE  types GLFW.MouseButton  GLFW.Action 
We get two overloads so we will be able to respond with single handler to both mouse click and mouse position
Enum GLFW.Action:
RELEASE = 0
PRESS = 1
REPEAT = 2
Enum GLFW.MouseButton:
MOUSE_BUTTON_1 = 0
MOUSE_BUTTON_2 = 1

experiments show that max x,y in window is both 600 if window width and height is 600 
 so in order to specify weather we are over aour quad we need to know how big is primary quad -
  defaoul it is occupying 100% of y axis and first left 80% of x axis
  hence we can calculate max height to equal the height of the window 
"""
@doc handlerStr
function (handler::MouseCallbackSubscribable)( a, x::Float64, y::Float64)
   # next!(handler.subject, CartesianIndex(Int(x),Int(y)))
   # if  (handler.isLeftButtonDown && x>=handler.xmin)
   if  (handler.isLeftButtonDown && x>=handler.xmin && x<=handler.xmax && y>=handler.ymin && y<= handler.ymax )
     #sending mouse position only if all conditions are met
     next!(handler.subject, CartesianIndex(Int(x),Int(y)))#sending mouse position only if all conditions are met
   else
    #next!(handler.subject, CartesianIndex(-1,-1))#sending mouse position only if all conditions are met
   end#if

end #handler

@doc handlerStr
function (handler::MouseCallbackSubscribable)(a, button::GLFW.MouseButton, action::GLFW.Action,m)
    handler.isLeftButtonDown = (button==GLFW.MOUSE_BUTTON_1 &&  action==GLFW.PRESS)#so it will stop either as we relese left mouse or click right
end #second handler



registerMouseClickFunctionsStr="""
we pass coordinate of cursor only when isLeftButtonDown is true and we make it true 
if left button is presed down - we make it true if the left button is pressed over image and false if mouse get out of the window or we get information about button release
imageWidth adn imageHeight are the dimensions of textures that we use to display 
"""
@doc registerMouseClickFunctionsStr
function registerMouseClickFunctions(window::GLFW.Window
                                    ,stopListening::Base.Threads.Atomic{Bool}
                                    )


 stopListening[]=true # stoping event listening loop to free the GLFW context

  # calculating dimensions of quad becouse it do not occupy whole window
    windowDims =     GLFW.GetWindowSize(window)
  width = windowDims[1]
  height = windowDims[2]
  quadmaxX = Int32(floor(width*0.8))
  quadMaxY = height 

  buttonSubs = MouseCallbackSubscribable(false,0,0,quadmaxX,quadMaxY,
  Subject((CartesianIndex{2}), scheduler = AsyncScheduler()))


# GLFW.SetScrollCallback(window, (a, xoff, yoff) -> scrollback(a, xoff, yoff))
GLFW.SetCursorPosCallback(window, (a, x, y) -> buttonSubs(a,x, y ) )# and  for example : cursor: 29.0, 469.0  types   Float64  Float64   
GLFW.SetMouseButtonCallback(window, (a, button, action, mods) ->buttonSubs(a,button, action,mods )) # for example types MOUSE_BUTTON_1 PRESS   GLFW.MouseButton  GLFW.Action 

stopListening[]=false # reactivate event listening loop

#subscription = subscribe!(buttonSubs, (direction) -> println(direction)) -usefull for debugging

return buttonSubs

end #registerMouseScrollFunctions


reactToScrollStr = """
in case of the scroll p true will be send in case of down - false
in response to it it sets new screen int variable and changes displayed screen
"""
@doc reactToScrollStr
function reactToMouseDrag(mouseCoord::CartesianIndex{2}, actor::SyncActor{Any, ActorWithOpenGlObjects})
    actor.actor.mainForDisplayObjects.stopListening[]=true #free GLFW context
    



    actor.actor.mainForDisplayObjects.stopListening[]=false # reactivete event listening loop


end#reactToScroll






end #ReactOnMouseClickAndDrag
