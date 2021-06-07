using DrWatson
@quickactivate "Probabilistic medical segmentation"
using Distributed
###### number of processes
addprocs(2)
## getting id of workers 
dirToWorkerNumbs = DrWatson.scriptsdir("mainPipeline","processesDefinitions","workerNumbers.jl")
include(dirToWorkerNumbs)



dirToH = DrWatson.scriptsdir("loadData","manageH5File.jl")
dirToWorkerPrepare = DrWatson.scriptsdir("mainPipeline","processesDefinitions","mainProcessDefinition.jl")
dirToFileStructs = DrWatson.scriptsdir("structs","forFilesEtc.jl")
include(dirToFileStructs)

include(dirToWorkerPrepare)
include(dirToFileStructs)
include(dirToH)

using Main.h5manag
@async initializeWorker(persistenceWorker, ["Documenter", "HDF5","ColorTypes"], [filePathAndModuleName(dirToH,"h5manag")] )

## image viewer helper

dirToImageHelper = DrWatson.scriptsdir("display","imageViewerHelper.jl")
@async initializeWorker(imageViewerHelperNumb, ["Documenter"], [filePathAndModuleName(dirToImageHelper,"imageViewerHelper")] )

