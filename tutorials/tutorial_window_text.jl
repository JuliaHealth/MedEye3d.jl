using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

using MedEye3d
using MedEye3d.DataStructs
using MedEye3d.BasicStructs
using MedEye3d.SegmentationDisplay
using MedEye3d.StructsManag
using ColorTypes
using Statistics

"""
Tutorial: Configuring Window Size and Amount of Space Allocated to Text

This script demonstrates:
1. Setting explicit window dimensions.
2. Allocating space for information text using fractionOfMainIm.
3. Displaying metrics (e.g., from MedEval) in the text area.
4. Loading and displaying the Spleen dataset from Medical Decathlon.
"""

# ============================================================================
# 1. LOADING SPLEEN DATASET FROM MEDICAL DECATHLON
# ============================================================================

println("Tutorial: Window Size and Text Configuration")
println("="^50)

# Paths to Spleen dataset - update these to match your local paths
spleenImagePath = "path/to/spleen_image.nii.gz"
spleenLabelPath = "path/to/spleen_label.nii.gz"

# Check if files exist
using FilePathsBase

if isfile(spleenImagePath) && isfile(spleenLabelPath)
    println("Loading Spleen dataset from Medical Decathlon...")
    
    # Load the registered images
    medImages = MedEye3d.SegmentationDisplay.loadRegisteredImages([
        (spleenImagePath, "CT"),
        (spleenLabelPath, "CT")
    ])
    
    # Extract data
    ctData = medImages[1].voxel_data
    labelData = medImages[2].voxel_data
    spacing = medImages[1].spacing
    
    println("  CT data size: ", size(ctData))
    println("  Label data size: ", size(labelData))
    println("  Voxel spacing: ", spacing)
    
else
    println("Spleen dataset not found at specified paths.")
    println("Using dummy data for demonstration.")
    println("To use real data, download from: http://medicaldecathlon.com/")
    println("and update the file paths above.")
    
    # Create dummy data for demonstration
    ctData = rand(Float32, 100, 100, 50)
    labelData = rand(0:1, 100, 100, 50)
    spacing = (1.0, 1.0, 1.0)
end

# ============================================================================
# 2. CONFIGURE TEXT TO DISPLAY WITH MEDEVAL METRICS
# ============================================================================

# Example metrics from MedEval evaluation
metrics = ResultMetrics(
    dice=0.87,
    jaccard=0.77,
    Hausdorff=3.8,
    precision=0.89,
    recall=0.85
)

# Global text (stays visible for all slices)
mainText = [
    SimpleLineTextStruct(text="Spleen Segmentation Analysis", fontSize=140),
    SimpleLineTextStruct(text="Dataset: Medical Decathlon (Spleen)", fontSize=100),
    SimpleLineTextStruct(text="", fontSize=80),
    SimpleLineTextStruct(text="EVALUATION METRICS:", fontSize=120),
    SimpleLineTextStruct(text="Dice Coefficient: $(round(metrics.dice, digits=3))"),
    SimpleLineTextStruct(text="Jaccard Index: $(round(metrics.jaccard, digits=3))"),
    SimpleLineTextStruct(text="Hausdorff Distance: $(round(metrics.Hausdorff, digits=2)) mm"),
    SimpleLineTextStruct(text="Precision: $(round(metrics.precision, digits=3))"),
    SimpleLineTextStruct(text="Recall: $(round(metrics.recall, digits=3))"),
    SimpleLineTextStruct(text="", fontSize=80),
    SimpleLineTextStruct(text="CONTROLS:", fontSize=120),
    SimpleLineTextStruct(text="Mouse Scroll: Navigate slices"),
    SimpleLineTextStruct(text="Right Click + Drag: Adjust windowing"),
    SimpleLineTextStruct(text="F1-F3: Preset window levels"),
    SimpleLineTextStruct(text="Space + 1/2/3: Change view plane")
]

# Slice-specific text (updates as you scroll)
sliceText = [
    [
        SimpleLineTextStruct(text="Slice: $i / $(size(ctData, 3))", fontSize=100),
        SimpleLineTextStruct(text="Intensity: Mean=$(round(mean(ctData[:, :, i]), digits=2))"),
        SimpleLineTextStruct(text="Labeled voxels: $(sum(labelData[:, :, i]))")
    ]
    for i in 1:size(ctData, 3)
]

# ============================================================================
# 3. PREPARE TEXTURE SPECIFICATIONS AND DIMENSIONS
# ============================================================================

# Texture specifications define how each data layer is displayed
textureSpecs = [
    # Main CT image
    TextureSpec{Float32}(
        name="CT",
        numb=Int32(3),
        isMainImage=true,
        color=RGB(1.0, 1.0, 1.0),
        minAndMaxValue=Float32.([-150, 250])  # Soft tissue window
    ),
    
    # Spleen segmentation mask
    TextureSpec{Float32}(
        name="Spleen_Mask",
        numb=Int32(1),
        color=RGB(1.0, 0.0, 0.0),  # Red for segmentation
        minAndMaxValue=Float32.([0, 1])
    ),
    
    # Manual modification mask (editable)
    TextureSpec{Float32}(
        name="Manual_Modif",
        numb=Int32(2),
        color=RGB(0.0, 1.0, 0.0),  # Green for manual edits
        minAndMaxValue=Float32.([0, 1]),
        isEditable=true
    )
]

# Metadata about dimensions
scrollDims = DataToScrollDims(
    imageSize=size(ctData),
    voxelSize=spacing,
    dimensionToScroll=3  # Scroll through axial slices
)

# ============================================================================
# 4. INITIALIZE VISUALIZER WITH WINDOW CONFIGURATION
# ============================================================================

println("\nWindow Configuration:")
println("  Width: 1200 pixels")
println("  Height: 800 pixels")
println("  Image space: 70% (fractionOfMainIm = 0.7)")
println("  Text space: 30%")
println("\nLaunching visualization...")

# Initialize visualizer with custom window size
# fractionOfMainIm = 0.7 means 70% width for image, 30% for text
mainMedEye3dInstance = coordinateDisplay(
    textureSpecs,
    0.7f0;  # Allocation: 70% image, 30% text
    dataToScrollDims=scrollDims,
    windowWidth=1200,
    windowHeight=800
)

# ============================================================================
# 5. PREPARE AND PASS DATA FOR SCROLLING
# ============================================================================

# Prepare data tuple vector
tupleVector = [
    ("CT", ctData),
    ("Spleen_Mask", Float32.(labelData)),
    ("Manual_Modif", zeros(Float32, size(ctData)))
]

# Get three-dimensional data structure
slicesData = StructsManag.getThreeDims(tupleVector)

# Create scrollable data structure
scrollData = FullScrollableDat(
    dataToScrollDims=scrollDims,
    dataToScroll=slicesData,
    mainTextToDisp=mainText,
    sliceTextToDisp=sliceText,
    segmMetr=metrics,  # Include metrics
    slicesNumber=Int32(size(ctData, 3))
)

# Pass data to the visualizer
passDataForScrolling(mainMedEye3dInstance, scrollData)

# ============================================================================
# 6. ADDITIONAL CONFIGURATION EXAMPLES
# ============================================================================

println("\n" * "="^50)
println("ADDITIONAL CONFIGURATION EXAMPLES:")
println("="^50)

println("\n1. Different window sizes:")
println("   coordinateDisplay(textureSpecs, 0.8f0, scrollDims, windowWidth=1000)")
println("   coordinateDisplay(textureSpecs, 0.8f0, scrollDims, windowWidth=800, windowHeight=600)")
println("   coordinateDisplay(textureSpecs, 0.8f0, scrollDims, windowWidth=1400, windowHeight=900)")

println("\n2. Different text space allocations:")
println("   coordinateDisplay(textureSpecs, 0.9f0, scrollDims)  # 90% image, 10% text")
println("   coordinateDisplay(textureSpecs, 0.8f0, scrollDims)  # 80% image, 20% text (default)")
println("   coordinateDisplay(textureSpecs, 0.6f0, scrollDims)  # 60% image, 40% text")
println("   coordinateDisplay(textureSpecs, 0.5f0, scrollDims)  # 50% image, 50% text")

println("\n3. Automatic height calculation:")
println("   coordinateDisplay(textureSpecs, 0.8f0, scrollDims, windowWidth=1000)")
println("   # Height automatically calculated as: 1000 * 0.8 = 800")

println("\n" * "="^50)
println("Visualizer started. Close the window to exit.")
println("\nTry adjusting the window configuration parameters to see different layouts.")
println("For more information, see the MedEye3d documentation.")