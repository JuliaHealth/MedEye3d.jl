module MedEye3d
import Logging
using Logging
using Base.Threads

export  ForDisplayStructs
# export  ForDisplayStructs.TextureSpec
export  SegmentationDisplay

# export  DataStructs.ThreeDimRawDat
# export  DataStructs.DataToScrollDims
# export  DataStructs.FullScrollableDat
# export  ForDisplayStructs.KeyboardStruct
# export  ForDisplayStructs.MouseStruct
# export  ForDisplayStructs.ActorWithOpenGlObjects
export  OpenGLDisplayUtils
# export  DisplayWords.textLinesFromStrings
export  StructsManag
# export  DisplayWords.textLinesFromStrings
# export  StructsManag.getThreeDims

include(joinpath( "display","GLFW","startModules","ModernGlUtil.jl"))

include(joinpath( "structs","BasicStructs.jl"))
include(joinpath( "structs","DataStructs.jl"))
include(joinpath( "structs","ForDisplayStructs.jl"))
include(joinpath( "structs","distinctColorsSaved.jl"))

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

include(joinpath( "display","reactingToMouseKeyboard","reactToKeyboard","KeyboardMouseHelper.jl") )

include(joinpath( "display","reactingToMouseKeyboard","reactToKeyboard","MaskDiffrence.jl") )
include(joinpath( "display","reactingToMouseKeyboard","reactToKeyboard","KeyboardVisibility.jl") )
include(joinpath( "display","reactingToMouseKeyboard","reactToKeyboard","OtherKeyboardActions.jl") )
include(joinpath( "display","reactingToMouseKeyboard","reactToKeyboard","WindowControll.jl") )
include(joinpath( "display","reactingToMouseKeyboard","reactToKeyboard","ChangePlane.jl") )
include(joinpath( "display","reactingToMouseKeyboard","reactToKeyboard","reactToKeyboard.jl") )



include(joinpath( "display","reactingToMouseKeyboard","ReactingToInput.jl") )

include(joinpath( "display","GLFW","SegmentationDisplay.jl"))
include(joinpath( "higherAbstractions","visualizationFromHdf5.jl"))

#checking weather we have some interactive thread reserved
if(VERSION < v"1.8")
    @error "MedEye3D above version 0.5.6 requires at least Julia 1.9 and setting of the interactive Thread (feature available from Julia 1.9 ) one can set it in linux by enviromental variable export JULIA_NUM_THREADS=3,1 where 1 after the coma is the interactive thread and 3 is the number of the other threads available on your machine; or start julia like this julia --threads 3,1; you can also use the docker container prepared by the author from  https://github.com/jakubMitura14/MedPipe3DTutorial. More about interactive THreads on https://docs.julialang.org/en/v1/manual/multi-threading/"
    throw(error())
end #if


# using Pkg
# ENV["MODERNGL_DEBUGGING"] = "true"
# Pkg.build("ModernGL")

greet() = print("Hello from medEye")

end # module
