
using DrWatson
@quickactivate "Probabilistic medical segmentation"

# include(DrWatson.scriptsdir("loadData","manageH5File.jl"))
# include(DrWatson.scriptsdir("display","mainDisplay.jl"))

# singleCtScanDisplay( getExample())

workersConfigDir = DrWatson.scriptsdir("mainPipeline","processesDefinitions","workersConfig.jl")
include(workersConfigDir)



######### just testing

exmpleH = @spawnat 2 Main.h5manag.getExample()

fetch(exmpleH)