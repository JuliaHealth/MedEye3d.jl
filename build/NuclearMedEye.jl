module NuclearMedEye
import Logging


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
include(joinpath( "display","reactingToMouseKeyboard","reactToKeyboard.jl") )

include(joinpath( "display","reactingToMouseKeyboard","ReactingToInput.jl") )
include(joinpath( "display","GLFW","SegmentationDisplay.jl"))


greet() = print("Hello World!")

end # module
