module MedEye3d
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



# Registration pull request created: JuliaRegistries/General/43883

# After the above pull request is merged, it is recommended that a tag is created on this repository for the registered package version.

# This will be done automatically if the Julia TagBot GitHub Action is installed, or can be done manually through the github interface, or via:

# git tag -a v0.1.0 -m "<description of version>" adf9b527e95ae0de193efdf9e14fd48fc045210f
# git push origin v0.1.0