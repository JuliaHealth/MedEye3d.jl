using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

using MedEye3d
using MedEye3d.DataStructs
using MedEye3d.BasicStructs
using MedEye3d.SegmentationDisplay
using ColorTypes

"""
Tutorial: Configuring Window Size and Text Display in MedEye3d

This tutorial demonstrates:
1. How to configure window dimensions (width and height)
2. How to allocate space between image and text using fractionOfMainIm
3. How to display text including metrics from MedEval
4. How to load and display data from the Medical Decathlon Spleen dataset

Prerequisites:
- Install MedEye3d.jl
- Download the Spleen dataset from Medical Decathlon (http://medicaldecathlon.com/)
"""

# ============================================================================
# 1. WINDOW SIZE CONFIGURATION
# ============================================================================

"""
Window size in MedEye3d is controlled through two main parameters in the 
coordinateDisplay function:

1. windowWidth::Int - Sets the width of the GLFW window in pixels
2. windowHeight::Int - Sets the height of the window (defaults to windowWidth * fractionOfMainIm)

You can also control the aspect ratio by setting both parameters explicitly.
"""

# Example 1: Default window size (1000px width, auto-calculated height)
# coordinateDisplay(textureSpecs, 0.8f0, dataToScrollDims, windowWidth=1000)

# Example 2: Custom window dimensions
# coordinateDisplay(textureSpecs, 0.8f0, dataToScrollDims, windowWidth=1200, windowHeight=800)

# Example 3: Square window
# coordinateDisplay(textureSpecs, 0.8f0, dataToScrollDims, windowWidth=800, windowHeight=800)

# ============================================================================
# 2. TEXT SPACE ALLOCATION
# ============================================================================

"""
The fractionOfMainIm parameter controls how much of the window width is allocated
to the main image versus the text panel:

- fractionOfMainIm = 0.8  -> 80% image, 20% text
- fractionOfMainIm = 0.7  -> 70% image, 30% text  
- fractionOfMainIm = 0.9  -> 90% image, 10% text

The text panel appears on the right side of the window and displays:
- Global text (visible on all slices)
- Slice-specific text (changes with scrolling)
"""

# ============================================================================
# 3. LOADING SPLEEN DATASET FROM MEDICAL DECATHLON
# ============================================================================

"""
Note: To run this example, you need to download the Spleen dataset from:
http://medicaldecathlon.com/

Place the files in a directory and update the paths below.
"""

# Paths to Spleen dataset (update these to match your local paths)
spleenImagePath = "path/to/spleen_image.nii.gz"
spleenLabelPath = "path/to/spleen_label.nii.gz"

# Check if files exist, otherwise use dummy data for demonstration
using FilePathsBase

if isfile(spleenImagePath) && isfile(spleenLabelPath)
    println("Loading Spleen dataset from Medical Decathlon...")
    
    # Load the registered images
    medImages = MedEye3d.SegmentationDisplay.loadRegisteredImages([
        (spleenImagePath, "CT"),
        (spleenLabelPath, "CT")
    ])
    
    # Extract data from loaded images
    ctData = medImages[1].voxel_data
    labelData = medImages[2].voxel_data
    
    # Get spacing information
    spacing = medImages[1].spacing
    
    println("Dataset loaded successfully:")
    println("  CT data size: ", size(ctData))
    println("  Label data size: ", size(labelData))
    println("  Voxel spacing: ", spacing)
    
else
    println("Spleen dataset not found. Using dummy data for demonstration.")
    println("To use real data, download from: http://medicaldecathlon.com/")
    
    # Create dummy data for demonstration
    ctData = rand(Float32, 100, 100, 50)
    labelData = rand(0:1, 100, 100, 50)
    spacing = (1.0, 1.0, 1.0)
end

# ============================================================================
# 4. CONFIGURING TEXT DISPLAY WITH MEDEVAL METRICS
# ============================================================================

"""
MedEval metrics can be displayed in the text panel using the ResultMetrics struct.
This is useful for showing segmentation evaluation results alongside the images.
"""

# Example metrics (in a real scenario, these would come from MedEval evaluation)
metrics = ResultMetrics(
    dice=0.87,
    jaccard=0.77,
    Hausdorff=3.8,
    precision=0.89,
    recall=0.85
)

# Global text - visible on all slices
mainText = [
    SimpleLineTextStruct(text="Spleen Segmentation Analysis", fontSize=140),
    SimpleLineTextStruct(text="Dataset: Medical Decathlon - Spleen", fontSize=100),
    SimpleLineTextStruct(text="", fontSize=80),  # Empty line for spacing
    SimpleLineTextStruct(text="EVALUATION METRICS:", fontSize=120),
    SimpleLineTextStruct(text="Dice Coefficient: $(round(metrics.dice, digits=3))"),
    SimpleLineTextStruct(text="Jaccard Index: $(round(metrics.jaccard, digits=3))"),
    SimpleLineTextStruct(text="Hausdorff Distance: $(round(metrics.Hausdorff, digits=2)) mm"),
    SimpleLineTextStruct(text="Precision: $(round(metrics.precision, digits=3))"),
    SimpleLineTextStruct(text="Recall: $(round(metrics.recall, digits=3))"),
    SimpleLineTextStruct(text="", fontSize=80),  # Empty line for spacing
    SimpleLineTextStruct(text="CONTROLS:", fontSize=120),
    SimpleLineTextStruct(text="Mouse Scroll: Navigate slices"),
    SimpleLineTextStruct(text="Right Click + Drag: Adjust windowing"),
    SimpleLineTextStruct(text="F1-F3: Preset window levels"),
    SimpleLineTextStruct(text="Space + 1/2/3: Change view plane")
]

# Slice-specific text - changes with scrolling
sliceText = [
    [
        SimpleLineTextStruct(text="Slice: $i / $(size(ctData, 3))", fontSize=100),
        SimpleLineTextStruct(text="Intensity Stats:", fontSize=90),
        SimpleLineTextStruct(text="  Mean: $(round(mean(ctData[:, :, i]), digits=2))"),
        SimpleLineTextStruct(text="  Std: $(round(std(ctData[:, :, i]), digits=2))"),
        SimpleLineTextStruct(text="Label Stats:", fontSize=90),
        SimpleLineTextStruct(text="  Voxels: $(sum(labelData[:, :, i]))")
    ]
    for i in 1:size(ctData, 3)
]

# ============================================================================
# 5. TEXTURE SPECIFICATION
# ============================================================================

"""
Texture specifications define how each data layer is displayed:
- Colors, visibility, windowing levels, etc.
"""

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
        name="Spleen_Segmentation",
        numb=Int32(1),
        color=RGB(1.0, 0.0, 0.0),  # Red for segmentation
        minAndMaxValue=Float32.([0, 1])
    ),
    
    # Manual modification mask (editable)
    TextureSpec{Float32}(
        name="Manual_Modifications",
        numb=Int32(2),
        color=RGB(0.0, 1.0, 0.0),  # Green for manual edits
        minAndMaxValue=Float32.([0, 1]),
        isEditable=true
    )
]

# ============================================================================
# 6. DATA STRUCTURES FOR DISPLAY
# ============================================================================

# Data dimensions and scrolling information
dataToScrollDims = DataToScrollDims(
    imageSize=size(ctData),
    voxelSize=spacing,
    dimensionToScroll=3  # Scroll through axial slices
)

# Prepare data for display
tupleVector = [
    ("CT", ctData),
    ("Spleen_Segmentation", Float32.(labelData)),
    ("Manual_Modifications", zeros(Float32, size(ctData)))
]

slicesData = StructsManag.getThreeDims(tupleVector)

# Main scrollable data structure
mainScrollData = FullScrollableDat(
    dataToScrollDims=dataToScrollDims,
    dimensionToScroll=1,  # Start with transverse view
    dataToScroll=slicesData,
    mainTextToDisp=mainText,
    sliceTextToDisp=sliceText,
    segmMetr=metrics  # Include metrics in the data structure
)

# ============================================================================
# 7. CONFIGURING AND LAUNCHING THE DISPLAY
# ============================================================================

println("\n" * "="^60)
println("CONFIGURATION SUMMARY:")
println("="^60)
println("Window width: 1200 pixels")
println("Window height: 800 pixels")
println("Image space: 70% of window (fractionOfMainIm = 0.7)")
println("Text space: 30% of window")
println("Text lines: $(length(mainText)) global, $(length(sliceText)) slice-specific")
println("Data size: $(size(ctData))")
println("="^60)
println("\nLaunching visualization...")

# Configure and launch the display with custom window size
try
    # Example 1: Custom window dimensions with 70% image space
    medEye3dInstance = coordinateDisplay(
        textureSpecs,
        0.7f0,  # 70% image, 30% text
        dataToScrollDims,
        windowWidth=1200,
        windowHeight=800
    )
    
    # Pass data for scrolling
    passDataForScrolling(medEye3dInstance, mainScrollData)
    
    println("\nVisualization launched successfully!")
    println("Close the window to exit.")
    println("\nTry these adjustments:")
    println("1. Change windowWidth to 1000 for smaller window")
    println("2. Change fractionOfMainIm to 0.8 for more image space")
    println("3. Change fractionOfMainIm to 0.6 for more text space")
    
catch e
    println("Error launching visualization: ", e)
    println("\nTroubleshooting tips:")
    println("1. Make sure OpenGL is available on your system")
    println("2. Check that all required packages are installed")
    println("3. Verify file paths if using real data")
end

# ============================================================================
# 8. ADDITIONAL CONFIGURATION EXAMPLES
# ============================================================================

"""
# Example: Different window configurations

# A. Wide window with more text space (for detailed reports)
wideWindowConfig = coordinateDisplay(
    textureSpecs,
    0.6f0,  # 60% image, 40% text
    dataToScrollDims,
    windowWidth=1400,
    windowHeight=900
)

# B. Compact window for quick viewing
compactConfig = coordinateDisplay(
    textureSpecs,
    0.85f0,  # 85% image, 15% text
    dataToScrollDims,
    windowWidth=800,
    windowHeight=600
)

# C. Square window for consistent aspect ratio
squareConfig = coordinateDisplay(
    textureSpecs,
    0.75f0,  # 75% image, 25% text
    dataToScrollDims,
    windowWidth=900,
    windowHeight=900
)

# D. Maximized text display for teaching/annotation
textHeavyConfig = coordinateDisplay(
    textureSpecs,
    0.5f0,  # 50% image, 50% text
    dataToScrollDims,
    windowWidth=1200,
    windowHeight=800
)
"""

# ============================================================================
# 9. BEST PRACTICES
# ============================================================================

"""
Best practices for window and text configuration:

1. Window Size:
   - 1200x800 is a good default for most displays
   - Consider user's screen resolution
   - Square windows (equal width/height) work well for consistent aspect ratios

2. Text Space Allocation:
   - Use 70-80% image space for primarily visual tasks
   - Use 50-60% image space when text information is critical
   - Adjust based on the amount of text to display

3. Text Content:
   - Keep global text concise (5-10 lines maximum)
   - Use slice-specific text for dynamic information
   - Format numbers for readability (2-3 decimal places)
   - Use empty lines (SimpleLineTextStruct(text="")) for spacing

4. Performance:
   - Larger windows require more GPU resources
   - More text lines increase rendering time
   - Consider window size when working with large datasets
"""

println("\n" * "="^60)
println("TUTORIAL COMPLETE")
println("="^60)
println("\nKey concepts demonstrated:")
println("1. Window size configuration via windowWidth/windowHeight")
println("2. Text space allocation via fractionOfMainIm")
println("3. Text display with SimpleLineTextStruct")
println("4. MedEval metrics integration with ResultMetrics")
println("5. Spleen dataset loading (when available)")
println("\nFor more information, see:")
println("- Medical Decathlon dataset: http://medicaldecathlon.com/")
println("- MedEye3d documentation: https://juliahealth.org/MedEye3d.jl/")