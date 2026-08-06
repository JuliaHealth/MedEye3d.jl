using MedEye3d
using MedEye3d.ForDisplayStructs
using MedEye3d.ReactOnMouseClickAndDrag

# Mock state
cDims = CalcDimsStruct(windowWidth=1000, windowHeight=1000, imageTextureWidth=100, imageTextureHeight=100)
mainStates = [StateDataFields(calcDimsStruct=cDims) for i in 1:5]
# Mock onScrollData and other fields so it doesn't crash on field access
for i in 1:5
    mainStates[i].switchIndex = 1
    mainStates[i].currentDisplayedSlice = 10
    
    # Mock data to scroll
    dummy_dat = rand(Float32, 100, 100, 50)
    mainStates[i].onScrollData = FullScrollableDat(
        dataToScroll = [ThreeDimRawDat(type=Float32, dat=dummy_dat)],
        dimensionToScroll = 3,
        slicesNumber = 50
    )
    
    # Set the quad verts properly
    mainStates[i].calcDimsStruct.mainImageQuadVert = Float32[
        -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ]
    # Mock texture specifications
    mainStates[i].mainForDisplayObjects = forDisplayObjects(
        listOfTextSpecifications = [TextureSpec{Float32}(name="image", numb=1)]
    )
end

# Test right click
try
    # Click on (500, 500) which should be center of screen.
    mousestr = MouseStruct(
        isRightButtonDown=true,
        lastCoordinates=[CartesianIndex(500, 500)],
        actualWindowWidth=1000,
        actualWindowHeight=1000
    )
    
    ReactOnMouseClickAndDrag.reactToMouseDrag(mousestr, mainStates)
    println("reactToMouseDrag right click SUCCEEDED with 5 panels!")
catch e
    println("reactToMouseDrag right click FAILED with 5 panels: ", e)
end

