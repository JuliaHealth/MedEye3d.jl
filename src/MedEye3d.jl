module MedEye3d
import Logging


export MedEye3d.ForDisplayStructs
export MedEye3d.ForDisplayStructs.TextureSpec
export MedEye3d.SegmentationDisplay

export MedEye3d.DataStructs.ThreeDimRawDat
export MedEye3d.DataStructs.DataToScrollDims
export MedEye3d.DataStructs.FullScrollableDat
export MedEye3d.ForDisplayStructs.KeyboardStruct
export MedEye3d.ForDisplayStructs.MouseStruct
export MedEye3d.ForDisplayStructs.ActorWithOpenGlObjects
export MedEye3d.OpenGLDisplayUtils
export MedEye3d.DisplayWords.textLinesFromStrings
export MedEye3d.StructsManag
export MedEye3d.DisplayWords.textLinesFromStrings
export MedEye3d.StructsManag.getThreeDims

include(joinpath( "display","GLFW","startModules","ModernGlUtil.jl"))

include(joinpath( "structs","BasicStructs.jl"))
include(joinpath( "structs","DataStructs.jl"))
include(joinpath( "structs","ForDisplayStructs.jl"))

include(joinpath( "display","GLFW","DispUtils","StructsManag.jl"))

include(joinpath( "display","GLFW","startModules","PrepareWindowHelpers.jl"))
include(joinpath( "display","GLFW","shadersEtc","CustomFragShad.jl"))

include(joinpath( "display","GLFW","DispUtils","OpenGLDisplayUtils.jl"))
include(joinpath( "display","GLFW","shadersEtc","ShadersAndVerticies.jl"))
include(joinpath( "display","GLFW","shadersEtc","ShadersAndVerticiesForText.jl"))
include(joinpath( "display","GLFW","shadersEtc","Uniforms.jl"))

include(joinpath( "display","GLFW","textRender","DisplayWords.jl"))

include(joinpath( "display","GLFW","DispUtils","TextureManag.jl") )
include(joinpath( "display","GLFW","startModules","PrepareWindow.jl"))



include(joinpath( "display","reactingToMouseKeyboard","ReactToScroll.jl") )
include(joinpath( "display","reactingToMouseKeyboard","ReactOnMouseClickAndDrag.jl") )


include(joinpath( "display","reactingToMouseKeyboard","reactToKeyboard","MaskDiffrence.jl") )
include(joinpath( "display","reactingToMouseKeyboard","reactToKeyboard","KeyboardVisibility.jl") )
include(joinpath( "display","reactingToMouseKeyboard","reactToKeyboard","OtherKeyboardActions.jl") )
include(joinpath( "display","reactingToMouseKeyboard","reactToKeyboard","WindowControll.jl") )
include(joinpath( "display","reactingToMouseKeyboard","reactToKeyboard","ChangePlane.jl") )
include(joinpath( "display","reactingToMouseKeyboard","reactToKeyboard","reactToKeyboard.jl") )

include(joinpath( "display","reactingToMouseKeyboard","ReactingToInput.jl") )
include(joinpath( "display","GLFW","SegmentationDisplay.jl"))


greet() = print("Hello World!")

end # module
