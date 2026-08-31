using Test
using MedEye3d
using MedEye3d.DataStructs
using MedEye3d.ForDisplayStructs
using MedEye3d.StructsManag
using MedEye3d.MakieEvents
using MedEye3d.ReactOnMouseClickAndDrag
using Dictionaries: Dictionary
using Setfield

const MEH = MedEye3d.SegmentationDisplay.MakieEventHandlers

@testset "Coordinate Mapping & Inversion Test" begin
    # 1400x900 window, QuadImage mode, top-left quadrant (panel 1)
    calcDim = CalcDimsStruct(
        windowWidth=1400,
        windowHeight=900,
        fractionOfMainIm=Float32(1.0),
        imageTextureWidth=512,
        imageTextureHeight=512,
        zoom=1.0f0,
        panX=0.0f0,
        panY=0.0f0
    )
    calcDim = StructsManag.getMainVerticies(calcDim, QuadImage, 1)
    
    actualW = 1400.0
    actualH = 900.0
    
    # Top-Left quadrant bounds: x in [0, 700], y in [0, 450]
    # In Vulkan:
    # At top-left corner of the image in panel 1:
    texX_top, texY_top = StructsManag.getTextureCoordinatesFromScreen(350, 10, calcDim, actualW, actualH)
    texX_mid, texY_mid = StructsManag.getTextureCoordinatesFromScreen(350, 225, calcDim, actualW, actualH)
    texX_bot, texY_bot = StructsManag.getTextureCoordinatesFromScreen(350, 440, calcDim, actualW, actualH)
    
    # Y coordinates should increase from top to bottom (texY_top < texY_mid < texY_bot)
    @test texY_top < texY_mid
    @test texY_mid < texY_bot
    @test texY_top >= 1
    @test texY_bot <= 512
    println("Coordinate mapping Y progression: top=$texY_top, mid=$texY_mid, bot=$texY_bot")
    
    # X coordinates should increase from left to right
    texX_left, _ = StructsManag.getTextureCoordinatesFromScreen(50, 225, calcDim, actualW, actualH)
    texX_center, _ = StructsManag.getTextureCoordinatesFromScreen(350, 225, calcDim, actualW, actualH)
    texX_right, _ = StructsManag.getTextureCoordinatesFromScreen(650, 225, calcDim, actualW, actualH)
    
    @test texX_left < texX_center
    @test texX_center < texX_right
    println("Coordinate mapping X progression: left=$texX_left, center=$texX_center, right=$texX_right")
end

@testset "Segmentation Painting & Dirty Flag Marking" begin
    w, h, d = 64, 64, 10
    ct_vol = zeros(Float32, w, h, d)
    mask_vol = zeros(Int16, w, h, d)
    
    textSpec_ct = TextureSpec{Float32}(name="CT", isMainImage=true)
    textSpec_mask = TextureSpec{Int16}(name="Mask", isMainImage=false, isMultiDiscreteMask=true, isEditable=true, strokeWidth=Int32(2))
    
    calcDim = CalcDimsStruct(windowWidth=800, windowHeight=800, imageTextureWidth=w, imageTextureHeight=h, zoom=1.0f0, panX=0.0f0, panY=0.0f0)
    calcDim = StructsManag.getMainVerticies(calcDim, SingleImage, 1)
    
    twoDim_ct = TwoDimRawDat{Float32}(Float32, "CT", selectdim(ct_vol, 3, 5))
    twoDim_mask = TwoDimRawDat{Int16}(Int16, "Mask", selectdim(mask_vol, 3, 5))
    
    twoDimList = [twoDim_ct, twoDim_mask]
    numbDict = Dictionary(["CT", "Mask"], [1, 2])
    
    singSl = SingleSliceDat(
        listOfDataAndImageNames=twoDimList,
        nameIndexes=numbDict,
        sliceNumber=5
    )
    
    forDisp = forDisplayObjects(
        listOfTextSpecifications=[textSpec_ct, textSpec_mask],
        TextureIndexes=numbDict
    )
    
    scrollDat = FullScrollableDat(
        dataToScroll=[ThreeDimRawDat{Float32}(Float32, "CT", ct_vol), ThreeDimRawDat{Int16}(Int16, "Mask", mask_vol)],
        dimensionToScroll=3,
        slicesNumber=d,
        dataToScrollDims=DataToScrollDims(imageSize=(w,h,d), voxelSize=(1.0,1.0,1.0), dimensionToScroll=3),
        nameIndexes=numbDict
    )
    
    state = StateDataFields(
        currentlyDispDat=singSl,
        mainForDisplayObjects=forDisp,
        calcDimsStruct=calcDim,
        onScrollData=scrollDat,
        textureToModifyVec=[textSpec_mask],
        valueForMasToSet=valueForMasToSetStruct(value=7, is_painting_active=true),
        currentDisplayedSlice=5,
        isSliceChanged=false
    )
    
    mainStates = [state]
    
    # Simulate mouse paint strokes across center of window
    m_stroke = [
        MouseStruct(isLeftButtonDown=true, lastCoordinates=[CartesianIndex(300, 400)], actualWindowWidth=800, actualWindowHeight=800),
        MouseStruct(isLeftButtonDown=true, lastCoordinates=[CartesianIndex(400, 400)], actualWindowWidth=800, actualWindowHeight=800),
        MouseStruct(isLeftButtonDown=true, lastCoordinates=[CartesianIndex(500, 400)], actualWindowWidth=800, actualWindowHeight=800)
    ]
    
    ReactOnMouseClickAndDrag.react_to_draw(m_stroke, mainStates)
    
    # 1. Verify slice is marked dirty for Vulkan upload
    @test state.isSliceChanged == true
    
    # 2. Verify all channels remain in currentlyDispDat
    @test length(state.currentlyDispDat.listOfDataAndImageNames) == 2
    @test state.currentlyDispDat.listOfDataAndImageNames[1].name == "CT"
    @test state.currentlyDispDat.listOfDataAndImageNames[2].name == "Mask"
    
    # 3. Verify voxels are set to label value 7 in 2D slice and 3D volume
    painted_voxels_2d = count(state.currentlyDispDat.listOfDataAndImageNames[2].dat .== Int16(7))
    painted_voxels_3d = count(mask_vol .== Int16(7))
    @test painted_voxels_2d > 0
    @test painted_voxels_3d == painted_voxels_2d
    println("Painted voxels: 2D=$painted_voxels_2d, 3D=$painted_voxels_3d (label=7)")
end

@testset "Lesion Moving & Coordinate Translation" begin
    w, h, d = 64, 64, 10
    ct_vol = zeros(Float32, w, h, d)
    mask_vol = zeros(Int16, w, h, d)
    
    # Create initial lesion at (20:25, 20:25, 5) with ID=3
    mask_vol[20:25, 20:25, 5] .= Int16(3)
    
    textSpec_ct = TextureSpec{Float32}(name="CT", isMainImage=true)
    textSpec_mask = TextureSpec{Int16}(name="Mask", isMainImage=false, isMultiDiscreteMask=true, isEditable=true, strokeWidth=Int32(2), minAndMaxValue=Int16.([3, 3]))
    
    calcDim = CalcDimsStruct(windowWidth=800, windowHeight=800, imageTextureWidth=w, imageTextureHeight=h, zoom=1.0f0, panX=0.0f0, panY=0.0f0)
    calcDim = StructsManag.getMainVerticies(calcDim, SingleImage, 1)
    
    twoDim_ct = TwoDimRawDat{Float32}(Float32, "CT", selectdim(ct_vol, 3, 5))
    twoDim_mask = TwoDimRawDat{Int16}(Int16, "Mask", selectdim(mask_vol, 3, 5))
    
    twoDimList = [twoDim_ct, twoDim_mask]
    numbDict = Dictionary(["CT", "Mask"], [1, 2])
    
    singSl = SingleSliceDat(
        listOfDataAndImageNames=twoDimList,
        nameIndexes=numbDict,
        sliceNumber=5
    )
    
    forDisp = forDisplayObjects(
        listOfTextSpecifications=[textSpec_ct, textSpec_mask],
        TextureIndexes=numbDict
    )
    
    scrollDat = FullScrollableDat(
        dataToScroll=[ThreeDimRawDat{Float32}(Float32, "CT", ct_vol), ThreeDimRawDat{Int16}(Int16, "Mask", mask_vol)],
        dimensionToScroll=3,
        slicesNumber=d,
        dataToScrollDims=DataToScrollDims(imageSize=(w,h,d), voxelSize=(1.0,1.0,1.0), dimensionToScroll=3),
        nameIndexes=numbDict
    )
    
    state = StateDataFields(
        currentlyDispDat=singSl,
        mainForDisplayObjects=forDisp,
        calcDimsStruct=calcDim,
        onScrollData=scrollDat,
        textureToModifyVec=[textSpec_mask],
        valueForMasToSet=valueForMasToSetStruct(value=3, is_painting_active=false),
        currentDisplayedSlice=5,
        isSliceChanged=false,
        moveLesionMode=true
    )
    mainStates = [state]
    
    # 1. Start moving lesion (Right button press)
    m_press = MouseStruct(isLeftButtonDown=false, isRightButtonDown=true, lastCoordinates=[CartesianIndex(250, 250)], actualWindowWidth=800, actualWindowHeight=800)
    ReactOnMouseClickAndDrag.reactToMouseDrag(m_press, mainStates)
    
    @test state.movingLesionID == 3
    @test length(state.movingLesionOriginalCoords) == 36
    
    # 2. Drag lesion (Right button drag with delta)
    m_drag = MouseStruct(isLeftButtonDown=false, isRightButtonDown=true, lastCoordinates=[CartesianIndex(350, 350)], actualWindowWidth=800, actualWindowHeight=800)
    ReactOnMouseClickAndDrag.reactToMouseDrag(m_drag, mainStates)
    
    # Verify lesion moved and dirty flag is set
    @test state.isSliceChanged == true
    @test count(mask_vol .== Int16(3)) == 36
    # Original coordinates (20, 20, 5) should now be 0
    @test mask_vol[20, 20, 5] == Int16(0)
    println("Move lesion successful: original region cleared, 36 voxels translated.")
end

@testset "Quad View & Panel 5 Compare Isolation" begin
    states = [StateDataFields() for _ in 1:5]
    for i in 1:5
        states[i].calcDimsStruct = CalcDimsStruct(windowWidth=1400, windowHeight=900, fractionOfMainIm=1.0f0, imageTextureWidth=512, imageTextureHeight=512)
        states[i].displayMode = QuadImage
    end
    
    # 1. Compare ON
    MEH.compare_mode[] = true
    MEH.updateQuadVertices!(states[1], :LeftHalf)
    MEH.updateQuadVertices!(states[5], :RightHalf)
    MEH.updateQuadVertices!(states[2], :Hidden)
    
    @test sum(abs.(states[5].calcDimsStruct.mainImageQuadVert)) > 0.01f0
    @test all(iszero, states[2].calcDimsStruct.mainImageQuadVert)
    
    # 2. Compare OFF
    MEH.compare_mode[] = false
    MEH.updateQuadVertices!(states[1], :TopLeft)
    MEH.updateQuadVertices!(states[2], :TopRight)
    MEH.updateQuadVertices!(states[5], :Hidden)
    for s in states; s.displayMode = QuadImage; end
    
    @test sum(abs.(states[2].calcDimsStruct.mainImageQuadVert)) > 0.01f0
    @test all(iszero, states[5].calcDimsStruct.mainImageQuadVert)
    
    # 3. Simulate ChangePlaneEvent while Compare Mode is OFF
    states[5].onScrollData.dataToScrollDims = DataToScrollDims(imageSize=(512,512,100), voxelSize=(1.0,1.0,1.0), dimensionToScroll=3)
    MEH.reactToChangePlane(ChangePlaneEvent(:Coronal), states)
    
    # Panel 5 must remain strictly HIDDEN
    @test all(iszero, states[5].calcDimsStruct.mainImageQuadVert)
    # Panel 2 must remain visible
    @test sum(abs.(states[2].calcDimsStruct.mainImageQuadVert)) > 0.01f0
    println("Quad View panel isolation verified: Panel 5 is strictly hidden, Panel 2 is visible.")
end
