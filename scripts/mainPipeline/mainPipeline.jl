
using DrWatson
@quickactivate "Probabilistic medical segmentation"

include(DrWatson.scriptsdir("loadData","manageH5File.jl"))
include(DrWatson.scriptsdir("display","mainDisplay.jl"))

singleCtScanDisplay( getExample())



