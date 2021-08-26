using DrWatson
@quickactivate "NuclearEye"

using Revise

includet(DrWatson.scriptsdir("loadData","DicomManage.jl"))


includet(DrWatson.scriptsdir("display","GLFW","startModules","ModernGlUtil.jl"))

includet(DrWatson.scriptsdir("structs","FromSegmentationEvaluation.jl"))
includet(DrWatson.scriptsdir("structs","DataStructs.jl"))
includet(DrWatson.scriptsdir("structs","forDisplayStructs.jl"))

includet(DrWatson.scriptsdir("display","GLFW","DispUtils","StructsManag.jl"))
includet(DrWatson.scriptsdir("loadData","manageH5File.jl"))

includet(DrWatson.scriptsdir("display","GLFW","startModules","PrepareWindowHelpers.jl"))
includet(DrWatson.scriptsdir("display","GLFW","shadersEtc","CustomFragShad.jl"))

includet(DrWatson.scriptsdir("display","GLFW","DispUtils","OpenGLDisplayUtils.jl"))
includet(DrWatson.scriptsdir("display","GLFW","shadersEtc","ShadersAndVerticies.jl"))
includet(DrWatson.scriptsdir("display","GLFW","shadersEtc","ShadersAndVerticiesForText.jl"))
includet(DrWatson.scriptsdir("display","GLFW","shadersEtc","Uniforms.jl"))

includet(DrWatson.scriptsdir("display","GLFW","textRender","DisplayWords.jl"))

includet(DrWatson.scriptsdir("display","GLFW","DispUtils","TextureManag.jl") )
includet(DrWatson.scriptsdir("display","GLFW","startModules","PrepareWindow.jl"))



includet(DrWatson.scriptsdir("display","reactingToMouseKeyboard","ReactToScroll.jl") )
includet(DrWatson.scriptsdir("display","reactingToMouseKeyboard","ReactOnMouseClickAndDrag.jl") )
includet(DrWatson.scriptsdir("display","reactingToMouseKeyboard","reactToKeyboard.jl") )

includet(DrWatson.scriptsdir("display","reactingToMouseKeyboard","ReactingToInput.jl") )
includet(DrWatson.scriptsdir("display","GLFW","SegmentationDisplay.jl"))
