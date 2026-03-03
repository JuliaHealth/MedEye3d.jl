using MedEye3d
using MedEye3d.DataStructs
using MedEye3d.BasicStructs
using MedEye3d.SegmentationDisplay
using ColorTypes

"""
Tutorial: Configuring Window Size and Amount of Space Allocated to Text

This script demonstrates:
1. Setting explicit window dimensions.
2. Allocating space for information text using `fractionOfMainIm`.
3. Displaying metrics (e.g., from MedEval) in the text area.
"""

# 1. Create dummy 3D data (100x100x50)
# In a real scenario, you would load the Spleen dataset like this:
# medImage = loadRegisteredImages(("path/to/spleen_image.nii.gz", "path/to/spleen_label.nii.gz"))
# data = medImage[1].voxel_data
data = rand(Float32, 100, 100, 50)
label = rand(0:1, 100, 100, 50)

# 2. Configure Text to Display
# Let's assume these metrics were calculated by MedEval
metrics = ResultMetrics(dice=0.82, jaccard=0.71, Hausdorff=4.5)

# Global text (stays visible for all slices)
mainText = [
    SimpleLineTextStruct(text="Spleen Segmentation Analysis", fontSize=140),
    SimpleLineTextStruct(text="---------------------------", fontSize=100),
    SimpleLineTextStruct(text="Dataset: Medical Decathlon (Spleen)"),
    SimpleLineTextStruct(text="Global Dice: $(metrics.dice)"),
    SimpleLineTextStruct(text="Global Jaccard: $(metrics.jaccard)"),
    SimpleLineTextStruct(text="Max Hausdorff: $(metrics.Hausdorff) mm"),
    SimpleLineTextStruct(text=""),
    SimpleLineTextStruct(text="Controls:", fontSize=120),
    SimpleLineTextStruct(text="Scroll: Change slice"),
    SimpleLineTextStruct(text="Right Click: Windowing")
]

# Slice-specific text (updates as you scroll)
sliceText = [
    [SimpleLineTextStruct(text="Slice: $i / 50"), SimpleLineTextStruct(text="Local Intensity: $(round(rand(), digits=2))")] 
    for i in 1:50
]

# 3. Prepare Texture and Dimensions
# We define how the images should be colored
textureSpecs = [
    TextureSpec{Float32}(name="CT", color=RGB(1.0, 1.0, 1.0), minAndMaxValue=[0.0, 1.1]),
    TextureSpec{Float32}(name="Spleen_Mask", color=RGB(1.0, 0.0, 0.0), minAndMaxValue=[0.0, 1.1])
]

# metadata about dimensions
scrollDims = DataToScrollDims(
    imageSize = (100, 100, 50),
    voxelSize = (1.0, 1.0, 1.0),
    dimensionToScroll = 3
)

# 4. Initialize Visualizer
# fractionOfMainIm = 0.7 means 70% width for image, 30% for text
mainMedEye3dInstance = coordinateDisplay(
    textureSpecs,
    0.7f0; # Allocation: 70% image, 30% text
    dataToScrollDims = scrollDims,
    windowWidth = 1200,
    windowHeight = 800
)

# 5. Pass Data for Scrolling
scrollData = FullScrollableDat(
    dataToScrollDims = scrollDims,
    dataToScroll = [
        ThreeDimRawDat{Float32}(name="CT", dat=data),
        ThreeDimRawDat{Float32}(name="Spleen_Mask", dat=Float32.(label))
    ],
    mainTextToDisp = mainText,
    sliceTextToDisp = sliceText,
    slicesNumber = 50
)

passDataForScrolling(mainMedEye3dInstance, scrollData)

println("Visualizer started. Close the window to exit.")
# In a REPL, the window will stay open. In a script execution, we might need a signal wait.
