

"""
module 
code adapted from https://discourse.julialang.org/t/custom-subject-in-rocket-jl-for-mouse-events-from-glfw/65133/3
it is design to help processing data from 
    -GLFW.SetCursorPosCallback(window, (_, x, y) -> println("cursor: x, y")) and  for example : cursor: 29.0, 469.0  types   Float64  Float64   
    -GLFW.SetMouseButtonCallback(window, (_, button, action, mods) -> println("button action"))  for example types MOUSE_BUTTON_1 PRESS   GLFW.MouseButton  GLFW.Action 
The main function is to mark the interaction of the mouse to be saved in appropriate mask and be rendered onto the screen
so we modify the data that is the basis of the mouse interaction mask  and we pass the data on so appropriate part of the texture would be modified to be displayed on screen

"""
module ReactOnMouseClickAndDrag
using Parameters, Rocket, GLFW, ModernGL,  ..ForDisplayStructs, ..TextureManag,  ..OpenGLDisplayUtils
using  Dates, Parameters,  ..DataStructs,  ..StructsManag
export registerMouseClickFunctions
export reactToMouseDrag

mouseClickHandler= nothing # used only in case of the benchmarking to simulate user interaction

"""
struct that enables reacting to  the input  from mouse click  and drag the input will be 
    Cartesian index represening (x,y)
     x and y position  of the mouse - will be recorded only if left mouse button is pressed or keep presssed
"""
@with_kw  mutable struct MouseCallbackSubscribable <: Subscribable{MouseStruct}
    #true if left button is presed down - we make it true if the left button is pressed over image and false if mouse get out of the window or we get information about button release
    isLeftButtonDown ::Bool 
    isRightButtonDown ::Bool 
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
    isBusy::Base.Threads.Atomic{Bool} # indicate to check weather the system is ready to receive the input

   
    subject :: Subject{MouseStruct} # coordinates of mouse 
   
end



"""
configuting Rocket on Subscribe so we get custom handler of input as we see we still need to define actor
"""
# function Rocket.on_subscribe!(handler::MouseCallbackSubscribable, actor::SyncActor{Any, ActorWithOpenGlObjects})
#     return subscribe!(handler.subject, actor)
# end

function Rocket.on_subscribe!(handler::MouseCallbackSubscribable, actor::SyncActor{Any, ActorWithOpenGlObjects})

    return subscribe!(handler.subject, actor)
end

"""
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
function (handler::MouseCallbackSubscribable)( a, x::Float64, y::Float64)
  point = CartesianIndex(Int(x),Int(y))
  handler.lastCoordinate = point

  if  (handler.isLeftButtonDown && x>=handler.xmin && x<=handler.xmax && y>=handler.ymin && y<= handler.ymax )
    push!(handler.coordinatesStoreForLeftClicks,point) # putting coordinate to the list it will be processed when context will be ready
        if(!handler.isBusy[])  

            #sending mouse position only if all conditions are met
            next!(handler.subject,MouseStruct( 
                isLeftButtonDown=handler.isLeftButtonDown
                ,isRightButtonDown=handler.isRightButtonDown
               ,lastCoordinates=  handler.coordinatesStoreForLeftClicks))#sending mouse position only if all conditions are met
            handler.coordinatesStoreForLeftClicks = [point]
        end#if
    #now we also get intrested about  right clicks becouse they help to controll 
    elseif(handler.isRightButtonDown)  
        next!(handler.subject,MouseStruct( 
            isLeftButtonDown=handler.isLeftButtonDown
            ,isRightButtonDown=handler.isRightButtonDown
           ,lastCoordinates= [point] ))#sending mouse position only if all conditions are met

   end#if

end #handler

function (handler::MouseCallbackSubscribable)(a, button::GLFW.MouseButton, action::GLFW.Action,m)
    

    res=(button==GLFW.MOUSE_BUTTON_1 &&  action==GLFW.PRESS)#so it will stop either as we relese left mouse or click right
    handler.isLeftButtonDown = res
    handler.isRightButtonDown = (button==GLFW.MOUSE_BUTTON_2 &&  action==GLFW.PRESS)

    if(res)
        handler.coordinatesStoreForLeftClicks = [handler.lastCoordinate]
    end #if
end #second handler



"""
we pass coordinate of cursor only when isLeftButtonDown is true and we make it true 
if left button is presed down - we make it true if the left button is pressed over image and false if mouse get out of the window or we get information about button release
imageWidth adn imageHeight are the dimensions of textures that we use to display 
"""
function registerMouseClickFunctions(window::GLFW.Window
                                    ,stopListening::Base.Threads.Atomic{Bool}
                                    ,calcD::CalcDimsStruct
                                    ,isBusy::Base.Threads.Atomic{Bool}     )


 stopListening[]=true # stoping event listening loop to free the GLFW context


 # calculating dimensions of quad becouse it do not occupy whole window, and we want to react only to those mouse positions that are on main image quad
  mouseButtonSubs = MouseCallbackSubscribable(isLeftButtonDown=false
                                        ,isRightButtonDown=false
                                        ,xmin=Int32(calcD.windowWidthCorr)
                                        ,ymin=Int32(calcD.windowHeightCorr)
                                        ,xmax=Int32(calcD.avWindWidtForMain-calcD.windowWidthCorr)
                                        ,ymax=Int32(calcD.avWindHeightForMain-calcD.windowHeightCorr)
                                        ,coordinatesStoreForLeftClicks=[]
                                        ,lastCoordinate=CartesianIndex(1,1)
                                        ,isBusy=isBusy
                                        ,subject=Subject(MouseStruct  ,scheduler = AsyncScheduler()))

mouseClickHandler=mouseButtonSubs

GLFW.SetCursorPosCallback(window, (a, x, y) -> mouseButtonSubs(a,x, y ) )# and  for example : cursor: 29.0, 469.0  types   Float64  Float64   
GLFW.SetMouseButtonCallback(window, (a, button, action, mods) ->mouseButtonSubs(a,button, action,mods )) # for example types MOUSE_BUTTON_1 PRESS   GLFW.MouseButton  GLFW.Action 

stopListening[]=false # reactivate event listening loop

return mouseButtonSubs



end #registerMouseScrollFunctions


 """
we use mouse coordinate to modify the texture that is currently active for modifications 
    - we take information about texture currently active for modifications from variables stored in actor
    from texture specification we take also its id and its properties ...
"""
function reactToMouseDrag(mousestr::MouseStruct, actor::SyncActor{Any, ActorWithOpenGlObjects})
    obj = actor.actor.mainForDisplayObjects
    obj.stopListening[]=true #free GLFW context
    actor.actor.isBusy[]=true# mark that OpenGL is busy
    textureList = actor.actor.textureToModifyVec
    mouseCoords= mousestr.lastCoordinates
    if (!isempty(textureList)  && mousestr.isLeftButtonDown && textureList[1].isEditable)
        texture= textureList[1]
        calcDim =  actor.actor.calcDimsStruct
        # two dimensional coordinates on plane of intrest (current slice)
       mappedCoords =  translateMouseToTexture(texture.strokeWidth
                                                ,mouseCoords
                                                ,actor.actor.calcDimsStruct)

       twoDimDat= actor.actor.currentlyDispDat|> # accessing currently displayed data
       (singSl)-> singSl.listOfDataAndImageNames[singSl.nameIndexes[texture.name]] #accessing the texture data we want to modify
      
       toSet =  convert( parameter_type(texture) ,actor.actor.valueForMasToSet.value)
       modSlice!(twoDimDat, mappedCoords, convert(twoDimDat.type, toSet ))|> # modifying data associated with texture
       (sliceDat)-> updateTexture(twoDimDat.type,sliceDat, texture,0,0,calcDim.imageTextureWidth,calcDim.imageTextureHeight  )

       glClear(GL_COLOR_BUFFER_BIT)
       basicRender(obj.window)
        #to enable undoing we just set the point we modified back to 0 
        addToforUndoVector(actor, ()-> begin
        modSlice!(twoDimDat, mappedCoords,  convert(twoDimDat.type, 0 ))|> # modifying data associated with texture
        (sliceDat)-> updateTexture(twoDimDat.type,sliceDat, texture,0,0,calcDim.imageTextureWidth,calcDim.imageTextureHeight  )
         basicRender(obj.window)
      
    end)

    elseif( mousestr.isRightButtonDown  )
        #we save data about right click position in order to change the slicing plane accordingly
        mappedCoords =  translateMouseToTexture(Int32(1)
        ,mouseCoords
        ,actor.actor.calcDimsStruct)

        actor.actor.lastRecordedMousePosition = cartTwoToThree(actor.actor.onScrollData.dataToScrollDims 
                                                                ,actor.actor.currentDisplayedSlice
                                                               ,mappedCoords[1]    )   
    end #if 
    actor.actor.isBusy[]=false # we can do sth in opengl
    obj.stopListening[]=false # reactivete event listening loop
end#..ReactToScroll


"""
given list of cartesian coordinates and some window/ image characteristics - it translates mouse positions
to cartesian coordinates of the texture
strokeWidth - the property connected to the texture marking how thick should be the brush
mouseCoords - list of coordinates of mouse positions while left button remains pressed
calcDims - set of values usefull for calculating mouse position
return vector of translated cartesian coordinates
"""
function translateMouseToTexture(strokeWidth::Int32
                                ,mouseCoords::Vector{CartesianIndex{2}}
                                ,calcD::CalcDimsStruct )::Vector{CartesianIndex{2}}
  


    xmin=calcD.windowWidthCorr
    ymin=calcD.windowHeightCorr
    xmax=calcD.avWindWidtForMain-calcD.windowWidthCorr
    ymax=calcD.avWindHeightForMain-calcD.windowHeightCorr


    return    map(c->CartesianIndex( getNewX(c[1],calcD),  getNewY(c[2] ,calcD)) ,mouseCoords)  |>
    (x)->filter(it->it[1]>0 && it[2]>0 ,x)  |>      # we do not want to try access it in point 0 as julia is 1 indexed                 
   (filteredList) -> map(point-> addStrokeWidth(point,Int64(strokeWidth) ) , filteredList) |>  # adding some points around the point of choice so will be better visible
   (matrix)->reduce(vcat, matrix)# when we added some oints around we got list of lists so now we need to flatten it out
   unique |># we want only unique elements
   uniq-> filter(it-> it[1]>xmin && it[1]<xmax && it[2]>ymin && it[it]<ymax  ,uniq)    # as we add new points they may end up getting outside the texture; we need to filter those out
end #translateMouseToTexture

"""
adding the width to the stroke so we will be able to controll how thicly we are painting ...
"""
function addStrokeWidth(point::CartesianIndex{2},strokeW::Int64  )
    return CartesianIndices((-strokeW:strokeW,-strokeW:strokeW)) |> # set of cartesian indices that we will filter ot later
    list->list.+point|> # making coordinates around point of intrest
    added-> filter(x-> ( abs(point[1]- x[1]) + abs(x[2] -point[2]))<strokeW  ,added )# filtering to distant points
end#addStrokeWidth



"""
helper function for translateMouseToTexture
"""
function getNewX(x::Int,calcD::CalcDimsStruct)::Int
  # first we subtract windowWidthCorr as in window the image do not need to start at the begining  of the window
   return Int64(floor( ((x- calcD.windowWidthCorr)/(calcD.correCtedWindowQuadWidth))*calcD.imageTextureWidth))
end#getNewX

"""
helper function for translateMouseToTexture
"""
function getNewY(y::Int,calcD::CalcDimsStruct)::Int
    Int64(floor(  ((calcD.correCtedWindowQuadHeight-y+calcD.windowHeightCorr)/calcD.correCtedWindowQuadHeight)*calcD.imageTextureHeight)  )     
   
end#getNewY

end #ReactOnMouseClickAndDrag
