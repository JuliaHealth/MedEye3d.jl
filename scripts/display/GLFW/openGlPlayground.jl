
using DrWatson
@quickactivate "Probabilistic medical segmentation"

using GLFW: Window
using BenchmarkTools: minimum


dirToWorkerNumbs = DrWatson.scriptsdir("mainPipeline","processesDefinitions","workerNumbers.jl")
include(dirToWorkerNumbs)
using Main.workerNumbers
using Distributed


include("/home/jakub/JuliaProjects/Probabilistic-medical-segmentation/scripts/display/GLFW/modernGL/fromGlMakie.jl")
include("/home/jakub/JuliaProjects/Probabilistic-medical-segmentation/scripts/structs/forDisplayStructs.jl")
include("/home/jakub/JuliaProjects/Probabilistic-medical-segmentation/scripts/loadData/manageH5File.jl")



exampleDat = h5manag.getExample()
dims = size(exampleDat)

#prepared =  prepareForDisplayOfTransverses(exampleDat, dims)
datt=  modifyData(exampleDat,45)
window = displayAll(datt,dims[2],dims[3] )

GLFW.DestroyWindow(window)

# using Distributed
# using SharedArrays
# addprocs(2)


function waitPrn()
sleep(500)
print("ddd")
sleep(1000)
print("222")
end

@async waitPrn()

t = @task begin;
    while(true)
     sleep(15);
     println("done");
    end
    end
schedule(t)