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
using Parameters, Rocket, GLFW, ModernGL, Main.ForDisplayStructs,Main.TextureManag, Main.OpenGLDisplayUtils
using  Dates, Parameters, Main.DataStructs, Main.StructsManag
export registerMouseClickFunctions
export reactToMouseDrag

MouseCallbackSubscribableStr="""
struct that enables reacting to  the input  from mouse click  and drag the input will be 
    Cartesian index represening (x,y)
     x and y position  of the mouse - will be recorded only if left mouse button is pressed or keep presssed
"""
@doc MouseCallbackSubscribableStr
@with_kw  mutable struct MouseCallbackSubscribable <: Subscribable{Vector{CartesianIndex{2}}}
    #true if left button is presed down - we make it true if the left button is pressed over image and false if mouse get out of the window or we get information about button release
    isLeftButtonDown ::Bool 
    #coordinates marking 4 corners of 
    #the quad that displays our medical image with the masks
    xmin::Int32
    ymin::Int32
    xmax::Int32
    ymax::Int32
#used to draw left button lines (creating lines)
 #store of the cartesian coordinates that is used to batch actions 
#- so if mouse is moving rapidly we would store bunch of coordinates and then modify texture in batch
    coordinatesStoreForLeftClicks ::Vector{CartesianIndex{2}} 
    lastCoordinate::CartesianIndex{2}#generally when we draw lines we remove points from array above yet w need to leave last one in order to keep continuity of futher line


referenceInstance::DateTime# an instance from which we would calculate when to execute batch 
    
    subject :: Subject{Vector{CartesianIndex{2}}} # coordinates of mouse 
   
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
  point = CartesianIndex(Int(x),Int(y))
  handler.lastCoordinate = point
  
  
  if  (handler.isLeftButtonDown && x>=handler.xmin && x<=handler.xmax && y>=handler.ymin && y<= handler.ymax )
    push!(handler.coordinatesStoreForLeftClicks,point)
        if((Dates.now()-handler.referenceInstance).value>100)  
            #sending mouse position only if all conditions are met
            next!(handler.subject, handler.coordinatesStoreForLeftClicks)#sending mouse position only if all conditions are met
            handler.referenceInstance=Dates.now()
            handler.coordinatesStoreForLeftClicks = [point]
        end#if
   end#if

end #handler

@doc handlerStr
function (handler::MouseCallbackSubscribable)(a, button::GLFW.MouseButton, action::GLFW.Action,m)

    res=(button==GLFW.MOUSE_BUTTON_1 &&  action==GLFW.PRESS)#so it will stop either as we relese left mouse or click right
    handler.isLeftButtonDown = res
    if(res)
        handler.referenceInstance=Dates.now()
        handler.coordinatesStoreForLeftClicks = [handler.lastCoordinate]
    end #if
end #second handler



registerMouseClickFunctionsStr="""
we pass coordinate of cursor only when isLeftButtonDown is true and we make it true 
if left button is presed down - we make it true if the left button is pressed over image and false if mouse get out of the window or we get information about button release
imageWidth adn imageHeight are the dimensions of textures that we use to display 
"""
@doc registerMouseClickFunctionsStr
function registerMouseClickFunctions(window::GLFW.Window
                                    ,stopListening::Base.Threads.Atomic{Bool}
                                    ,calcD::CalcDimsStruct     )


 stopListening[]=true # stoping event listening loop to free the GLFW context


 # calculating dimensions of quad becouse it do not occupy whole window, and we want to react only to those mouse positions that are on main image quad
  mouseButtonSubs = MouseCallbackSubscribable(isLeftButtonDown=false
                                        ,xmin=Int32(calcD.windowWidthCorr)
                                        ,ymin=Int32(calcD.windowHeightCorr)
                                        ,xmax=Int32(calcD.avWindWidtForMain-calcD.windowWidthCorr)
                                        ,ymax=Int32(calcD.avWindHeightForMain-calcD.windowHeightCorr)
                                        ,coordinatesStoreForLeftClicks=Vector{CartesianIndex{2}}()
                                        ,lastCoordinate=CartesianIndex(1,1)
                                        ,referenceInstance=Dates.now()
                                        ,subject=Subject(Vector{CartesianIndex{2}}  ,scheduler = AsyncScheduler()))

GLFW.SetCursorPosCallback(window, (a, x, y) -> mouseButtonSubs(a,x, y ) )# and  for example : cursor: 29.0, 469.0  types   Float64  Float64   
GLFW.SetMouseButtonCallback(window, (a, button, action, mods) ->mouseButtonSubs(a,button, action,mods )) # for example types MOUSE_BUTTON_1 PRESS   GLFW.MouseButton  GLFW.Action 

stopListening[]=false # reactivate event listening loop

return mouseButtonSubs



end #registerMouseScrollFunctions


reactToMouseDragStr = """
we use mouse coordinate to modify the texture that is currently active for modifications 
    - we take information about texture currently active for modifications from variables stored in actor
    from texture specification we take also its id and its properties ...
"""
@doc reactToMouseDragStr
function reactToMouseDrag(mouseCoords::Vector{CartesianIndex{2}}, actor::SyncActor{Any, ActorWithOpenGlObjects})
    obj = actor.actor.mainForDisplayObjects
    obj.stopListening[]=true #free GLFW context
    textureList = actor.actor.textureToModifyVec

    if (!isempty(textureList))
        texture= textureList[1]
    

        # two dimensional coordinates on plane of intrest (current slice)
       mappedCoords =  translateMouseToTexture(texture.strokeWidth
                                                ,mouseCoords
                                                ,actor.actor.calcDimsStruct)
       twoDimDat= actor.actor.currentlyDispDat|> # accessing currently displayed data
       (singSl)-> singSl.listOfDataAndImageNames[singSl.nameIndexes[texture.name]] #accessing the texture data we want to modify
       
       modSlice!(twoDimDat, mappedCoords, convert(twoDimDat.type,1))|> # modifying data associated with texture
       (sliceDat)-> updateTexture(twoDimDat.type,sliceDat, texture)

        basicRender(obj.window)


    end #if 
    obj.stopListening[]=false # reactivete event listening loop
end#reactToScroll


```@doc
given list of cartesian coordinates and some window/ image characteristics - it translates mouse positions
to cartesian coordinates of the texture
strokeWidth - the property connected to the texture marking how thick should be the brush
mouseCoords - list of coordinates of mouse positions while left button remains pressed
calcDims - set of values usefull for calculating mouse position
return vector of translated cartesian coordinates
```
function translateMouseToTexture(strokeWidth::Int32
                                ,mouseCoords::Vector{CartesianIndex{2}}
                                ,calcD::CalcDimsStruct )
  

    halfStroke =   Int64(floor(strokeWidth/2 ))
    #updating given texture that we are intrested in in place we are intested in 

    return map(c->CartesianIndex( getNewX(c[1],calcD),  getNewY(c[2] ,calcD)) ,mouseCoords)  |>
             (x)->filter(it->it[1]>0 && it[2]>0 ,x)        # we do not want to try access it in point 0 as julia is 1 indexed                 

end #translateMouseToTexture

```@doc
helper function for translateMouseToTexture
```
function getNewX(x::Int,calcD::CalcDimsStruct)::Int
  # first we subtract windowWidthCorr as in window the image do not need to start at the begining  of the window
   return Int64(floor( ((x- calcD.windowWidthCorr)/(calcD.correCtedWindowQuadWidth))*calcD.imageTextureWidth))
end#getNewX

```@doc
helper function for translateMouseToTexture
```
function getNewY(y::Int,calcD::CalcDimsStruct)::Int
    Int64(floor(  ((calcD.correCtedWindowQuadHeight-y+calcD.windowHeightCorr)/calcD.correCtedWindowQuadHeight)*calcD.imageTextureHeight)  )     
   
end#getNewY

end #ReactOnMouseClickAndDrag
