

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
using Logging, Parameters, Setfield, GLFW, ModernGL, ..ForDisplayStructs, ..TextureManag, ..OpenGLDisplayUtils
using Dates, Parameters, ..DataStructs, ..StructsManag, Logging, Base.Threads
import Logging, Base.Threads
export registerMouseClickFunctions
export reactToMouseDrag
export react_to_draw



"""
we pass coordinate of cursor only when isLeftButtonDown is true and we make it true
if left button is presed down - we make it true if the left button is pressed over image and false if mouse get out of the window or we get information about button release
imageWidth adn imageHeight are the dimensions of textures that we use to display
"""
function registerMouseClickFunctions(window::GLFW.Window, calcD::CalcDimsStruct, mainChannel::Base.Channel{Any})
    xmin = Int32(calcD.windowWidthCorr)
    xmax = Int32(calcD.avWindWidtForMain - calcD.windowWidthCorr)

    ymin = Int32(calcD.windowHeightCorr)
    ymax = Int32(calcD.avWindHeightForMain - calcD.windowHeightCorr)
    # calculating dimensions of quad becouse it do not occupy whole window, and we want to react only to those mouse positions that are on main image quad
    mouseStructInstance = MouseStruct()

    GLFW.SetCursorPosCallback(window, (a, x, y) -> begin
        if (mouseStructInstance.isLeftButtonDown && x >= xmin && x <= xmax && y >= ymin && y <= ymax)
            point = CartesianIndex(Int(x), Int(y))
            mouseStructInstance.lastCoordinates = [point]
            put!(mainChannel, mouseStructInstance)

        end
    end)# and  for example : cursor: 29.0, 469.0  types   Float64  Float64
    GLFW.SetMouseButtonCallback(window, (a, button, action, mods) -> begin
        leftMouseButtonDownResult = (button == GLFW.MOUSE_BUTTON_1 && action == GLFW.PRESS)
        mouseStructInstance.isLeftButtonDown = leftMouseButtonDownResult

        rightMouseButtonDownResult = (button == GLFW.MOUSE_BUTTON_2 && action == GLFW.PRESS)
        mouseStructInstance.isRightButtonDown = rightMouseButtonDownResult

        # put!(mainChannel, mouseStructInstance)
    end) # for example types MOUSE_BUTTON_1 PRESS   GLFW.MouseButton  GLFW.Action

end #registerMouseScrollFunctions



mouseCoords_channel = Base.Channel{MouseStruct}(100)
# we can fetch! on the channel, what is the next thing line, if the mouseStruct, check previous one by fetch. If it mouseStruct, aggregate those 2 and fetch the next one
#fetch in while loop, until no more mouseStructs, then we have the last one, and we can react to it


"""
used when we want to save some manual modifications
"""
# function react_to_draw(textureList,actor,mouseCoords_channel)
function react_to_draw(mouseStructArray::Vector{MouseStruct}, stateObject)
    # sleep(0.1);
    # @info "react_to_draw after sleep" isready(mouseCoords_channel)
    texture = stateObject.textureToModifyVec[1]
    calcDim = stateObject.calcDimsStruct


    # mouseCoords=take!(mouseCoords_channel).lastCoordinates
    # mappedCoords=translateMouseToTexture(texture.strokeWidth, mouseCoords, actor.actor.calcDimsStruct)
    # # two dimensional coordinates on plane of intrest (current slice)

    """
    get a list of MouseStruct from fetch and take!
    map each MouseStruct using translateMouseToTexture
    result will be mappedCoords, in react_to_draw
    check whether the length of the mappedCoords is greater than 0
    """
    mappedCoords = Vector{CartesianIndex{2}}()
    for mouseStruct in mouseStructArray
        mouseCoords = mouseStruct.lastCoordinates
        append!(mappedCoords, translateMouseToTexture(texture.strokeWidth, mouseCoords, stateObject.calcDimsStruct))
    end
    # append!(mappedCoords, translateMouseToTexture(texture.strokeWidth, mouseCoords, stateObject.calcDimsStruct))

    # @info "react_to_draw after channel" mappedCoords

    # is_sth_in=true

    twoDimDat = stateObject.currentlyDispDat |> # accessing currently displayed data
                (singSl) -> singSl.listOfDataAndImageNames[singSl.nameIndexes[texture.name]] #accessing the texture data we want to modify



    toSet = convert(parameter_type(texture), stateObject.valueForMasToSet.value)
    sliceDat = modSlice!(twoDimDat, mappedCoords, convert(twoDimDat.type, toSet)) # modifying data associated with texture


    #  updateTexture(twoDimDat.type,sliceDat, texture,0,0,calcDim.imageTextureWidth,calcDim.imageTextureHeight  )

    singleSliceDat = setproperties(stateObject.currentlyDispDat, (listOfDataAndImageNames = [sliceDat]))



    updateImagesDisplayed(singleSliceDat, stateObject.mainForDisplayObjects, stateObject.textDispObj, stateObject.calcDimsStruct, stateObject.valueForMasToSet)
end#react_to_draw


"""
we use mouse coordinate to modify the texture that is currently active for modifications
    - we take information about texture currently active for modifications from variables stored in actor
    from texture specification we take also its id and its properties ...
"""
function reactToMouseDrag(mousestr::MouseStruct, mainState::StateDataFields)
    obj = mainState.mainForDisplayObjects
    textureList = mainState.textureToModifyVec
    mouseCoords = mousestr.lastCoordinates

    #we save data about right click position in order to change the slicing plane accordingly
    mappedCoords = translateMouseToTexture(Int32(1), mouseCoords, mainState.calcDimsStruct)
    mappedCorrd = mappedCoords
    if (!isempty(mappedCorrd))
        cartMapped = cartTwoToThree(mainState.onScrollData.dataToScrollDims, mainState.currentDisplayedSlice, mappedCoords[1])

        mainState.lastRecordedMousePosition = cartMapped
    end#if

end#..ReactToScroll


"""
given list of cartesian coordinates and some window/ image characteristics - it translates mouse positions
to cartesian coordinates of the texture
strokeWidth - the property connected to the texture marking how thick should be the brush
mouseCoords - list of coordinates of mouse positions while left button remains pressed
calcDims - set of values usefull for calculating mouse position
return vector of translated cartesian coordinates
"""
function translateMouseToTexture(strokeWidth::Int32, mouseCoords::Vector{CartesianIndex{2}}, calcD::CalcDimsStruct)::Vector{CartesianIndex{2}}


    filteredList = map(c -> CartesianIndex(getNewX(c[1], calcD), getNewY(c[2], calcD)), mouseCoords) |>
                   (x) -> filter(it -> it[1] > 0 && it[2] > 0, x)       # we do not want to try access it in point 0 as julia is 1 indexed
    if (!isempty(filteredList))
        return map(point -> addStrokeWidth(point, Int64(strokeWidth)), filteredList) |>  # adding some points around the point of choice so will be better visible
               (matrix) -> reduce(vcat, matrix) |># when we added some oints around we got list of lists so now we need to flatten it out
                           unique |> # we want only unique elements
                           uniq -> filter(it -> it[1] > 0 && it[1] < calcD.imageTextureWidth && it[2] > 0 && it[2] < calcD.imageTextureHeight, uniq)     # as we add new points they may end up getting outside the texture; we need to filter those out
    end #if
    #if we are here we do not have anything meaningfull else to return
    return Vector{CartesianIndex{2}}()
end #translateMouseToTexture

"""
adding the width to the stroke so we will be able to controll how thicly we are painting ...
"""
function addStrokeWidth(point::CartesianIndex{2}, strokeW::Int64)
    return CartesianIndices((-strokeW:strokeW, -strokeW:strokeW)) |> # set of cartesian indices that we will filter ot later
           list -> list .+ point |> # making coordinates around point of intrest
                   added -> filter(x -> (abs(point[1] - x[1]) + abs(x[2] - point[2])) < strokeW, added)# filtering to distant points
end#addStrokeWidth

xx = CartesianIndex(2, 2)
xx[1]
"""
helper function for translateMouseToTexture
"""
function getNewX(x::Int, calcD::CalcDimsStruct)::Int
    # first we subtract windowWidthCorr as in window the image do not need to start at the begining  of the window
    return Int64(floor(((x - calcD.windowWidthCorr) / (calcD.correCtedWindowQuadWidth)) * calcD.imageTextureWidth))
end#getNewX

"""
helper function for translateMouseToTexture
"""
function getNewY(y::Int, calcD::CalcDimsStruct)::Int
    Int64(floor(((calcD.correCtedWindowQuadHeight - y + calcD.windowHeightCorr) / calcD.correCtedWindowQuadHeight) * calcD.imageTextureHeight))

end#getNewY

end #ReactOnMouseClickAndDrag
