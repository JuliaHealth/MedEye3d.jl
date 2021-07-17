
using DrWatson
@quickactivate "Probabilistic medical segmentation"

using GLFW: Window
using BenchmarkTools: minimum

dirToWorkerNumbs = DrWatson.scriptsdir("mainPipeline","processesDefinitions","workerNumbers.jl")
include(dirToWorkerNumbs)
using Main.workerNumbers


include("/home/jakub/JuliaProjects/Probabilistic-medical-segmentation/scripts/display/GLFW/modernGL/fromGlMakie.jl")
include("/home/jakub/JuliaProjects/Probabilistic-medical-segmentation/scripts/structs/forDisplayStructs.jl")
include("/home/jakub/JuliaProjects/Probabilistic-medical-segmentation/scripts/loadData/manageH5File.jl")



exampleDat = h5manag.getExample()
dat = modifyData(exampleDat)
displayAll( dat[1],dat[2],dat[3] )


# rr= rand(Int32,2*2)
# displayAll(rr, 2,2)

#displayAll(Int16.([1,2,6,9]), 2,2)
# minn = -1024 ;
# maxx  = 3071;
# rang = 4095;

# min_shown_white = 360 
# max_shown_black = 50
# dispRAng = min_shown_white-max_shown_black



# function ter(x)
#     if x<max_shown_black
#         return 0
#     elseif x>min_shown_white
#         return 1
#     else
#         return (x-max_shown_black)/dispRAng
#     end

# end

# map(ter ,[-200,51 ,215,300, 359, 2000])

