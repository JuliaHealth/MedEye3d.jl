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
export OpenGLDisplayUtils
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

include(joinpath("preprocessing", "BoneSubsegmentation.jl"))
using .BoneSubsegmentation
export BoneSubsegmentation

include(joinpath("display", "LesionMetadataWindow.jl"))
include(joinpath("display", "HeuristicsEngine.jl"))
include(joinpath("display", "StudySelectorWindow.jl"))
using .StudySelectorWindow: open_study_selector, scan_medical_files
export open_study_selector, scan_medical_files, StudySelectorWindow

include(joinpath("packaging", "AppMain.jl"))
using .MedEye3dApp: julia_main
export julia_main

greet() = print("Hello from medEye")

end # module
