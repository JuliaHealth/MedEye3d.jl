# Tutorial: Window Size and Text Configuration in MedEye3d

## Overview

This tutorial explains how to configure window dimensions and text display in MedEye3d.jl, a 3D medical image visualization library. You'll learn how to control the size of the visualization window and allocate space between medical images and informational text.

## Prerequisites

- MedEye3d.jl installed (`Pkg.add("MedEye3d")`)
- Basic familiarity with Julia programming
- (Optional) Spleen dataset from Medical Decathlon for real data examples

## 1. Window Size Configuration

### Basic Window Parameters

MedEye3d provides two main parameters for window configuration:

```julia
# In the coordinateDisplay function:
medEye3dInstance = coordinateDisplay(
    textureSpecs,
    fractionOfMainIm,  # Controls image/text space allocation
    dataToScrollDims,
    windowWidth=1200,   # Window width in pixels
    windowHeight=800    # Window height in pixels
)
```

### Common Window Size Presets

```julia
# Standard desktop size (good for most applications)
coordinateDisplay(textureSpecs, 0.7f0, dataToScrollDims, windowWidth=1200, windowHeight=800)

# Compact window (for side-by-side viewing)
coordinateDisplay(textureSpecs, 0.8f0, dataToScrollDims, windowWidth=800, windowHeight=600)

# Large window (for presentations or detailed work)
coordinateDisplay(textureSpecs, 0.75f0, dataToScrollDims, windowWidth=1600, windowHeight=1200)

# Square window (consistent aspect ratio)
coordinateDisplay(textureSpecs, 0.7f0, dataToScrollDims, windowWidth=900, windowHeight=900)
```

### Automatic Height Calculation

If you don't specify `windowHeight`, it's automatically calculated based on the window width and `fractionOfMainIm`:

```julia
# Height = windowWidth * fractionOfMainIm
coordinateDisplay(textureSpecs, 0.8f0, dataToScrollDims, windowWidth=1000)
# Results in: windowWidth=1000, windowHeight=800
```

## 2. Text Space Allocation

### The `fractionOfMainIm` Parameter

The `fractionOfMainIm` parameter controls how much of the window width is allocated to the medical image versus the text panel:

| Value | Image Space | Text Space | Use Case |
|-------|-------------|------------|----------|
| 0.9 | 90% | 10% | Primarily visual tasks |
| 0.8 | 80% | 20% | Default balance |
| 0.7 | 70% | 30% | Detailed analysis with metrics |
| 0.6 | 60% | 40% | Teaching/annotation heavy |
| 0.5 | 50% | 50% | Maximum text display |

### Example Configurations

```julia
# Medical image focused (minimal text)
coordinateDisplay(textureSpecs, 0.9f0, dataToScrollDims, windowWidth=1200)

# Balanced view (default)
coordinateDisplay(textureSpecs, 0.8f0, dataToScrollDims, windowWidth=1200)

# Text-heavy for displaying MedEval metrics
coordinateDisplay(textureSpecs, 0.6f0, dataToScrollDims, windowWidth=1200)
```

## 3. Text Display Configuration

### Text Structure Types

MedEye3d uses two types of text displays:

1. **Global Text**: Visible on all slices (e.g., patient info, overall metrics)
2. **Slice-specific Text**: Changes with scrolling (e.g., slice number, local measurements)

### Creating Text Lines

```julia
using MedEye3d.ForDisplayStructs

# Global text (appears on all slices)
mainText = [
    SimpleLineTextStruct(text="Patient: ABC123", fontSize=120),
    SimpleLineTextStruct(text="Study: Abdominal CT", fontSize=100),
    SimpleLineTextStruct(text="Date: 2024-03-04", fontSize=100),
    SimpleLineTextStruct(text="", fontSize=80),  # Empty line for spacing
    SimpleLineTextStruct(text="Segmentation Metrics:", fontSize=120),
    SimpleLineTextStruct(text="Dice: 0.87", fontSize=100),
    SimpleLineTextStruct(text="Jaccard: 0.77", fontSize=100)
]

# Slice-specific text (one array per slice)
sliceText = [
    [
        SimpleLineTextStruct(text="Slice: $i/$(totalSlices)", fontSize=100),
        SimpleLineTextStruct(text="Mean Intensity: $(meanValue)", fontSize=90)
    ]
    for i in 1:totalSlices
]
```

### Text Formatting Options

```julia
SimpleLineTextStruct(
    text="Your text here",
    fontSize=120,           # Font size in points
    extraLineSpace=1.2      # Line spacing multiplier
)
```

## 4. Displaying MedEval Metrics

### Integrating ResultMetrics

MedEye3d includes a `ResultMetrics` struct for storing segmentation evaluation results:

```julia
using MedEye3d.BasicStructs

# Create metrics from MedEval evaluation
metrics = ResultMetrics(
    dice=0.87,
    jaccard=0.77,
    Hausdorff=3.8,
    precision=0.89,
    recall=0.85
)

# Convert metrics to display text
metricText = [
    SimpleLineTextStruct(text="EVALUATION RESULTS:", fontSize=140),
    SimpleLineTextStruct(text="Dice: $(round(metrics.dice, digits=3))"),
    SimpleLineTextStruct(text="Jaccard: $(round(metrics.jaccard, digits=3))"),
    SimpleLineTextStruct(text="Hausdorff: $(round(metrics.Hausdorff, digits=2)) mm"),
    SimpleLineTextStruct(text="Precision: $(round(metrics.precision, digits=3))"),
    SimpleLineTextStruct(text="Recall: $(round(metrics.recall, digits=3))")
]
```

### Complete Example with Spleen Dataset

```julia
using Pkg
Pkg.activate(".")
using MedEye3d
using MedEye3d.DataStructs
using MedEye3d.BasicStructs
using MedEye3d.SegmentationDisplay
using ColorTypes
using Statistics

# Load Spleen dataset (update paths for your system)
spleenImagePath = "path/to/spleen_1.nii.gz"
spleenLabelPath = "path/to/spleen_1_gt.nii.gz"

# Load data
medImages = MedEye3d.SegmentationDisplay.loadRegisteredImages([
    (spleenImagePath, "CT"),
    (spleenLabelPath, "CT")
])

ctData = medImages[1].voxel_data
labelData = medImages[2].voxel_data

# Create example metrics (in practice, calculate from MedEval)
metrics = ResultMetrics(dice=0.85, jaccard=0.74, Hausdorff=5.2)

# Configure text display
mainText = [
    SimpleLineTextStruct(text="Spleen Segmentation - Medical Decathlon", fontSize=140),
    SimpleLineTextStruct(text="", fontSize=80),
    SimpleLineTextStruct(text="Metrics:", fontSize=120),
    SimpleLineTextStruct(text="Dice: $(metrics.dice)"),
    SimpleLineTextStruct(text="Jaccard: $(metrics.jaccard)"),
    SimpleLineTextStruct(text="Hausdorff: $(metrics.Hausdorff) mm")
]

sliceText = [
    [SimpleLineTextStruct(text="Slice: $i/$(size(ctData, 3))")]
    for i in 1:size(ctData, 3)
]

# Texture specifications
textureSpecs = [
    TextureSpec{Float32}(
        name="CT",
        numb=Int32(3),
        isMainImage=true,
        minAndMaxValue=Float32.([-150, 250])
    ),
    TextureSpec{Float32}(
        name="Spleen_Seg",
        numb=Int32(1),
        color=RGB(1.0, 0.0, 0.0),
        minAndMaxValue=Float32.([0, 1])
    )
]

# Data dimensions
dataToScrollDims = DataToScrollDims(
    imageSize=size(ctData),
    voxelSize=medImages[1].spacing,
    dimensionToScroll=3
)

# Prepare data
tupleVector = [
    ("CT", ctData),
    ("Spleen_Seg", Float32.(labelData))
]

slicesData = StructsManag.getThreeDims(tupleVector)

mainScrollData = FullScrollableDat(
    dataToScrollDims=dataToScrollDims,
    dimensionToScroll=1,
    dataToScroll=slicesData,
    mainTextToDisp=mainText,
    sliceTextToDisp=sliceText,
    segmMetr=metrics
)

# Launch visualization with custom window size
medEye3dInstance = coordinateDisplay(
    textureSpecs,
    0.7f0,  # 70% image, 30% text
    dataToScrollDims,
    windowWidth=1200,
    windowHeight=800
)

passDataForScrolling(medEye3dInstance, mainScrollData)
```

## 5. Best Practices and Recommendations

### Window Size Guidelines

| Use Case | Recommended Size | fractionOfMainIm |
|----------|-----------------|------------------|
| Clinical review | 1200x800 | 0.8-0.9 |
| Research analysis | 1400x900 | 0.7-0.8 |
| Teaching/demos | 1600x1200 | 0.6-0.7 |
| Mobile/laptop | 800x600 | 0.8-0.85 |

### Text Display Tips

1. **Keep it concise**: Limit global text to 10-15 lines maximum
2. **Use formatting**: Vary font sizes for hierarchy (titles: 140, content: 100-120)
3. **Add spacing**: Use empty lines (`SimpleLineTextStruct(text="")`) to separate sections
4. **Dynamic content**: Use slice-specific text for measurements that change with position
5. **Format numbers**: Round to 2-3 decimal places for readability

### Performance Considerations

- Larger windows require more GPU memory
- More text lines increase rendering time
- Consider reducing window size for very large datasets
- Test different `fractionOfMainIm` values to balance visibility and performance

## 6. Troubleshooting

### Common Issues

1. **Window doesn't appear**: Check OpenGL installation and GPU drivers
2. **Text not displaying**: Verify font availability and text rendering setup
3. **Poor performance**: Reduce window size or text complexity
4. **Incorrect aspect ratio**: Set both `windowWidth` and `windowHeight` explicitly

### Debugging Tips

```julia
# Print configuration before launching
println("Window configuration:")
println("  Width: $windowWidth")
println("  Height: $windowHeight")
println("  Image space: $(fractionOfMainIm*100)%")
println("  Text space: $((1-fractionOfMainIm)*100)%")
println("  Text lines: $(length(mainText)) global, $(length(sliceText)) per slice")
```

## 7. Advanced Configuration

### Multi-monitor Setup

For multi-monitor configurations, you can position the window:

```julia
# Note: This requires additional GLFW configuration
# Currently, MedEye3d uses default GLFW window positioning
```

### High-DPI Displays

MedEye3d automatically handles high-DPI scaling on supported systems. For manual control:

```julia
# Adjust font sizes for high-DPI displays
highDPIText = [
    SimpleLineTextStruct(text="High DPI Text", fontSize=160),  # Larger for high DPI
    # ... more text lines
]
```

## 8. Summary

Configuring window size and text display in MedEye3d involves:

1. **Window dimensions**: Set via `windowWidth` and `windowHeight` parameters
2. **Space allocation**: Control with `fractionOfMainIm` (0.0-1.0)
3. **Text content**: Create using `SimpleLineTextStruct` arrays
4. **Metrics integration**: Use `ResultMetrics` for MedEval results
5. **Best practices**: Balance visual space, text information, and performance

By following this tutorial, you can create optimized visualization setups for various medical imaging tasks, from clinical review to research analysis and educational demonstrations.

## Additional Resources

- [Medical Decathlon Dataset](http://medicaldecathlon.com/)
- [MedEye3d Documentation](https://juliahealth.org/MedEye3d.jl/)
- [GLFW Window Guide](https://www.glfw.org/docs/latest/window_guide.html)
- [OpenGL Best Practices](https://www.khronos.org/opengl/wiki/Best_Practices)