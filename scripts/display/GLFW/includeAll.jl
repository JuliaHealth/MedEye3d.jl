using DrWatson
@quickactivate "Probabilistic medical segmentation"

include(DrWatson.scriptsdir("structs","FromSegmentationEvaluation.jl"))
include(DrWatson.scriptsdir("structs","DataStructs.jl"))
include(DrWatson.scriptsdir("structs","forDisplayStructs.jl"))

include(DrWatson.scriptsdir("loadData","StructsManag.jl"))


include(DrWatson.scriptsdir("display","GLFW","startModules","PrepareWindowHelpers.jl"))
include(DrWatson.scriptsdir("display","GLFW","shadersEtc","CustomFragShad.jl"))
include(DrWatson.scriptsdir("generalUtils","MultiDimArrUtil.jl"))

include(DrWatson.scriptsdir("display","GLFW","modernGL","OpenGLDisplayUtils.jl"))
include(DrWatson.scriptsdir("display","GLFW","shadersEtc","ShadersAndVerticies.jl"))
include(DrWatson.scriptsdir("display","GLFW","shadersEtc","ShadersAndVerticiesForText.jl"))
include(DrWatson.scriptsdir("display","GLFW","shadersEtc","Uniforms.jl"))


include(DrWatson.scriptsdir("display","GLFW","modernGL","TextureManag.jl") )
include(DrWatson.scriptsdir("display","GLFW","startModules","PrepareWindow.jl"))

include(DrWatson.scriptsdir("display","GLFW","textRender","DisplayWords.jl"))


include(DrWatson.scriptsdir("display","reactingToMouseKeyboard","ReactToScroll.jl") )
include(DrWatson.scriptsdir("display","reactingToMouseKeyboard","ReactOnMouseClickAndDrag.jl") )
include(DrWatson.scriptsdir("display","reactingToMouseKeyboard","reactToKeyboard.jl") )

include(DrWatson.scriptsdir("display","reactingToMouseKeyboard","ReactingToInput.jl") )
include(DrWatson.scriptsdir("display","GLFW","SegmentationDisplay.jl"))
