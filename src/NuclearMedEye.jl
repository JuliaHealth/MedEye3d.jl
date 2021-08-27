module NuclearMedEye

include(joinpath("src","loadData","DicomManage.jl"))


include(joinpath("src","loadData","DicomManage.jl"))


include(joinpath("src","display","GLFW","startModules","ModernGlUtil.jl"))

include(joinpath("src","structs","FromSegmentationEvaluation.jl"))
include(joinpath("src","structs","DataStructs.jl"))
include(joinpath("src","structs","forDisplayStructs.jl"))

include(joinpath("src","display","GLFW","DispUtils","StructsManag.jl"))
include(joinpath("src","loadData","manageH5File.jl"))

include(joinpath("src","display","GLFW","startModules","PrepareWindowHelpers.jl"))
include(joinpath("src","display","GLFW","shadersEtc","CustomFragShad.jl"))

include(joinpath("src","display","GLFW","DispUtils","OpenGLDisplayUtils.jl"))
include(joinpath("src","display","GLFW","shadersEtc","ShadersAndVerticies.jl"))
include(joinpath("src","display","GLFW","shadersEtc","ShadersAndVerticiesForText.jl"))
include(joinpath("src","display","GLFW","shadersEtc","Uniforms.jl"))

include(joinpath("src","display","GLFW","textRender","DisplayWords.jl"))

include(joinpath("src","display","GLFW","DispUtils","TextureManag.jl") )
include(joinpath("src","display","GLFW","startModules","PrepareWindow.jl"))



include(joinpath("src","display","reactingToMouseKeyboard","ReactToScroll.jl") )
include(joinpath("src","display","reactingToMouseKeyboard","ReactOnMouseClickAndDrag.jl") )
include(joinpath("src","display","reactingToMouseKeyboard","reactToKeyboard.jl") )

include(joinpath("src","display","reactingToMouseKeyboard","ReactingToInput.jl") )
include(joinpath("src","display","GLFW","SegmentationDisplay.jl"))

using SegmentationDisplay

greet() = print("Hello World!")

end # module
