# Configuring Window Size and Text Display

MedEye3d.jl allows for flexible configuration of the visualization window and a dedicated area for displaying text, such as metrics, metadata, or instructions.

## Quick Start Tutorial

For a complete tutorial with the Spleen dataset from Medical Decathlon, see:
- [Window Size and Text Configuration Tutorial](../tutorials/tutorial_window_text.jl)
- [Comprehensive Guide](./window_text_configuration_tutorial.md)

## Window Dimensions

The window size is configured using `windowWidth` and `windowHeight` parameters in the `coordinateDisplay` function (or `displayImage`). 

- `windowWidth`: Total width of the window in pixels.
- `windowHeight`: Total height of the window in pixels.

If `windowHeight` is not provided, it defaults to a value based on the `fractionOfMainIm`.

## Text Area Configuration

The space allocated to text is controlled by the `fractionOfMainIm` parameter.

- **`fractionOfMainIm` (Float32)**: This value (between 0.0 and 1.0) defines what fraction of the window width is dedicated to the main image display. The remaining space (`1 - fractionOfMainIm`) is used for the text panel.

For example, `fractionOfMainIm = 0.8` means 80% of the width is image, and 20% is text.

```julia
# Example: 1200x800 window with 30% space for text
mainMedEye3dInstance = coordinateDisplay(
    listOfTextSpecs,
    0.7f0; # 70% image, 30% text
    windowWidth = 1200,
    windowHeight = 800
)
```

## Adding Text to the View

Text is handled via `SimpleLineTextStruct` and passed to the visualizer as `mainTextToDisp` (global) or `sliceTextToDisp` (per-slice).

### `SimpleLineTextStruct`
Each line of text is defined by this struct:
- `text`: The string to display.
- `fontSize`: Size of the font (default ~110).
- `extraLineSpace`: Multiplier for vertical spacing.

### Example: Displaying MedEval Metrics

You can use the `ResultMetrics` struct to store evaluation results and display them.

```julia
using MedEye3d.DataStructs
using MedEye3d.BasicStructs

# 1. Define global metrics
metrics = ResultMetrics(dice=0.85, jaccard=0.74, Hausdorff=5.2)

# 2. Convert to lines of text
textLines = [
    SimpleLineTextStruct(text="Global Metrics:", fontSize=140),
    SimpleLineTextStruct(text="Dice: $(metrics.dice)"),
    SimpleLineTextStruct(text="Jaccard: $(metrics.jaccard)"),
    SimpleLineTextStruct(text="Hausdorff: $(metrics.Hausdorff) mm")
]

# 3. Pass to FullScrollableDat
scrollData = FullScrollableDat(
    dataToScroll = [...],
    mainTextToDisp = textLines,
    slicesNumber = 100
)
```

## Advanced: Per-Slice Text

To show different text for each slice (e.g., local Dice score), use `sliceTextToDisp`:

```julia
# Vector of vectors (one per slice)
sliceSpecificText = [
    [SimpleLineTextStruct(text="Slice $i Dice: $(rand())")] 
    for i in 1:100
]

scrollData = FullScrollableDat(
    # ...
    sliceTextToDisp = sliceSpecificText
)
```

## Best Practices

1. **Window Size**: Use 1200x800 for standard desktops, 800x600 for compact views
2. **Text Allocation**: Use 70-80% image space for visual tasks, 50-60% when text is critical
3. **Text Content**: Keep global text concise (5-10 lines), use slice-specific text for dynamic info
4. **Performance**: Larger windows require more GPU resources

## Complete Example with Spleen Dataset

See the [tutorial file](../tutorials/tutorial_window_text.jl) for a complete example using the Medical Decathlon Spleen dataset, including:
- Loading NIfTI files
- Configuring window size and text space
- Displaying MedEval metrics
- Setting up texture specifications for proper display
