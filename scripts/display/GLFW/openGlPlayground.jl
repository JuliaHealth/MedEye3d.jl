using BenchmarkTools: minimum
include("/home/jakub/JuliaProjects/Probabilistic-medical-segmentation/scripts/display/GLFW/modernGL/fromGlMakie.jl")
include("/home/jakub/JuliaProjects/Probabilistic-medical-segmentation/scripts/structs/forDisplayStructs.jl")
include("/home/jakub/JuliaProjects/Probabilistic-medical-segmentation/scripts/loadData/manageH5File.jl")

exampleDat = h5manag.getExample()
dat = modifyData(exampleDat)
displayAll( dat[1],dat[2],dat[3] )
