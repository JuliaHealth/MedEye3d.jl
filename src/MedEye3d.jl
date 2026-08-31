module MedEye3d
import Logging
using Logging
using Base.Threads

# Configure logging to save to data/log.txt automatically (falling back to temporary/null logger if read-only)
function __init__()
    try
        log_dir = joinpath(@__DIR__, "..", "data")
        if !isdir(log_dir)
            mkdir(log_dir)
        end
        global_logger(SimpleLogger(open(joinpath(log_dir, "log.txt"), "a")))
    catch
        # Read-only or packaged installation fallback
    end
end

export ForDisplayStructs
# export  ForDisplayStructs.TextureSpec
export SegmentationDisplay

# export  DataStructs.ThreeDimRawDat
# export  DataStructs.DataToScrollDims
# export  DataStructs.FullScrollableDat
# export  ForDisplayStructs.KeyboardStruct
# export  ForDisplayStructs.MouseStruct
# export  ForDisplayStructs.ActorWithOpenGlObjects
# export  DisplayWords.textLinesFromStrings
export StructsManag
export ShadersAndVerticiesForSuperVoxels
export LesionTracker
# export  DisplayWords.textLinesFromStrings
# export  StructsManag.getThreeDims

include(joinpath("display", "GLFW", "startModules", "ModernGlUtil.jl"))

include(joinpath("structs", "BasicStructs.jl"))
include(joinpath("structs", "DataStructs.jl"))
include(joinpath("structs", "ForDisplayStructs.jl"))
include(joinpath("structs", "MakieEvents.jl"))
using .MakieEvents
export MakieEvents
include(joinpath("structs", "distinctColorsSaved.jl"))

include(joinpath("display", "GLFW", "DispUtils", "StructsManag.jl"))

include(joinpath("display", "GLFW", "startModules", "PrepareWindowHelpers.jl"))
include(joinpath("display", "GLFW", "shadersEtc", "CustomFragShad.jl"))

include(joinpath("display", "GLFW", "DispUtils", "OpenGLDisplayUtils.jl"))
include(joinpath("display", "GLFW", "shadersEtc", "ShadersAndVerticies.jl"))
include(joinpath("display", "GLFW", "shadersEtc", "ShadersAndVerticiesForText.jl"))
include(joinpath("display", "GLFW", "shadersEtc", "Uniforms.jl"))
include(joinpath("display", "GLFW", "textRender", "DisplayWords.jl"))
include(joinpath("display", "GLFW", "DispUtils", "TextureManag.jl"))
include(joinpath("display", "GLFW", "shadersEtc", "ShadersAndVerticiesForLine.jl"))
include(joinpath("display", "GLFW", "shadersEtc", "ShadersAndVerticiesForSupervoxels.jl"))
include(joinpath("display", "GLFW", "startModules", "PrepareWindow.jl"))

# Vulkan rendering backend (alternative to OpenGL)
include(joinpath("display", "Vulkan", "VulkanBackend.jl"))
using .VulkanBackend
export VulkanBackend



include(joinpath("postprocessing", "StrokeRasterization.jl"))
using .StrokeRasterization
export StrokeRasterization

include(joinpath("postprocessing", "ConnectedComponents.jl"))
using .ConnectedComponents
export ConnectedComponents

include(joinpath("display", "reactingToMouseKeyboard", "ReactToScroll.jl"))
include(joinpath("display", "reactingToMouseKeyboard", "ReactOnMouseClickAndDrag.jl"))

include(joinpath("display", "reactingToMouseKeyboard", "reactToKeyboard", "KeyboardMouseHelper.jl"))

include(joinpath( "display","reactingToMouseKeyboard","reactToKeyboard","MaskDiffrence.jl") )
include(joinpath( "display","reactingToMouseKeyboard","reactToKeyboard","KeyboardVisibility.jl") )
include(joinpath( "display","reactingToMouseKeyboard","reactToKeyboard","OtherKeyboardActions.jl") )
include(joinpath( "display","reactingToMouseKeyboard","reactToKeyboard","WindowControll.jl") )
include(joinpath( "display","reactingToMouseKeyboard","reactToKeyboard","ChangePlane.jl") )
include(joinpath( "display","reactingToMouseKeyboard","reactToKeyboard","ReactToKeyboard.jl") )

include(joinpath("display", "reactingToMouseKeyboard", "ReactingToInput.jl"))
include(joinpath("display", "InferenceClient.jl"))
using .InferenceClient
include(joinpath("display", "LesionAssociation.jl"))
using .LesionAssociation
include("LesionTracker.jl")
include(joinpath("higherAbstractions", "DisplayDataManag.jl"))
include(joinpath("display", "GLFW", "SegmentationDisplay.jl"))
include(joinpath("ai", "AIInference.jl"))
using .AIInference
export AIInference


# BoneSubsegmentation removed — bone surface/marrow data is purely precomputed
# in preprocessing (recompute_bones.jl) using max_anatomy + Skellytour.
# Loaded from Bone_Subsegments_0.h5 at startup into bone_subsegments_cache.


include(joinpath("display", "LesionMetadataWindow.jl"))
include(joinpath("display", "HeuristicsEngine.jl"))
include(joinpath("display", "StudySelectorWindow.jl"))
using .StudySelectorWindow: open_study_selector, scan_medical_files
export open_study_selector, scan_medical_files, StudySelectorWindow

include(joinpath("packaging", "AppMain.jl"))
using .MedEye3dApp: julia_main
export julia_main

greet() = print("Hello from medEye")

# ─── Precompilation Workload ───────────────────────────────────────────
# Precompile painting, rasterization, coordinate mapping, and event handling
# so first-use latency for drawing strokes is virtually zero.
let
    try
        # 1. StrokeRasterization CPU kernels for common mask data types
        for T in (Int16, Float32, UInt8, Int8, Int32)
            m = zeros(T, 64, 64)
            StrokeRasterization.rasterize_polyline!(m, [(10, 10), (20, 20), (30, 25)], 3, T(1))
            StrokeRasterization.rasterize_thick_line!(m, (5, 5), (15, 15), 2, T(2))
            
            # Slices of 3D arrays (SubArray views)
            vol = zeros(T, 64, 64, 4)
            view_2d = view(vol, :, :, 2)
            StrokeRasterization.rasterize_polyline!(view_2d, [(10, 10), (20, 20)], 3, T(1))
            StrokeRasterization.rasterize_thick_line!(view_2d, (5, 5), (15, 15), 2, T(2))
        end
        
        # 2. Coordinate conversions and texture coordinate mapping
        calc_dim = DataStructs.CalcDimsStruct(
            windowWidth=1000, windowHeight=1000,
            imageTextureWidth=512, imageTextureHeight=512,
            mainImageQuadVert=Float32[-1.0, -1.0, 1.0, 1.0],
            zoom=1.0f0, panX=0.0f0, panY=0.0f0
        )
        StructsManag.getTextureCoordinatesFromScreen(500.0, 500.0, calc_dim, 1000.0, 1000.0)
        
        # 3. Interactive painting dispatch state
        spec_mask = ForDisplayStructs.TextureSpec{Int16}(name="Mask", isEditable=true, minAndMaxValue=[1, 1000])
        spec_ct = ForDisplayStructs.TextureSpec{Float32}(name="CT", isEditable=false, minAndMaxValue=[-150, 250])
        
        two_dim_mask = DataStructs.TwoDimRawDat{Int16}(type=Int16, name="Mask", dat=zeros(Int16, 64, 64))
        disp_dat = DataStructs.SingleSliceDat(
            listOfDataAndImageNames=[two_dim_mask],
            nameIndexes=DataStructs.getLocationDict([two_dim_mask])
        )
        
        st = ForDisplayStructs.StateDataFields(
            switchIndex=1,
            mainForDisplayObjects=ForDisplayStructs.forDisplayObjects(listOfTextSpecifications=[spec_mask, spec_ct]),
            calcDimsStruct=calc_dim,
            valueForMasToSet=ForDisplayStructs.valueForMasToSetStruct(value=1, is_painting_active=true),
            textureToModifyVec=[spec_mask],
            currentlyDispDat=disp_dat,
            onScrollData=DataStructs.FullScrollableDat(
                dataToScroll=[
                    DataStructs.ThreeDimRawDat{Float32}(type=Float32, name="CT", dat=zeros(Float32, 64, 64, 8)),
                    DataStructs.ThreeDimRawDat{Int16}(type=Int16, name="Mask", dat=zeros(Int16, 64, 64, 8))
                ],
                dimensionToScroll=3,
                slicesNumber=Int32(8)
            )
        )
        
        mouse_struct = ForDisplayStructs.MouseStruct(
            lastCoordinates=[CartesianIndex(500, 500)],
            isLeftButtonDown=true,
            actualWindowWidth=1000,
            actualWindowHeight=1000
        )
        
        # Precompile draw handler and event handlers
        ReactOnMouseClickAndDrag.react_to_draw([mouse_struct], [st])
        SegmentationDisplay.MakieEventHandlers.reactToPaintVal(MakieEvents.PaintValEvent(1, true), [st])
        SegmentationDisplay.MakieEventHandlers.reactToChangeBrushSize(MakieEvents.ChangeBrushSizeEvent(3), [st])
        SegmentationDisplay.MakieEventHandlers.reactToShowSingleLesion(MakieEvents.ShowSingleLesionEvent(0), [st])
        SegmentationDisplay.MakieEventHandlers.reactToShowSingleLesion(MakieEvents.ShowSingleLesionEvent(1), [st])
        SegmentationDisplay.MakieEventHandlers.reactToSyncLesion(MakieEvents.SyncLesionEvent(1), [st])
    catch e
        @debug "Precompilation workload note: $e"
    end
end

end # module
