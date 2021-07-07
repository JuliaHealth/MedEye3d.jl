using DrWatson
@quickactivate "Probabilistic medical segmentation"
using Distributed
 
###### number of processes
addprocs(1)
## getting id of workers 
dirToWorkerNumbs = DrWatson.scriptsdir("mainPipeline","processesDefinitions","workerNumbers.jl")
include(dirToWorkerNumbs)
using GLMakie
using ColorSchemeTools
 
using Main.workerNumbers
@everywhere using Pkg
@everywhere Pkg.add("Parameters")
@everywhere Pkg.add("Documenter")
@everywhere Pkg.add("DrWatson")


@everywhere using Parameters
@everywhere using Documenter
@everywhere using DrWatson



dirToFileStructs = DrWatson.scriptsdir("structs","forFilesEtc.jl")
dirToImageStructs = DrWatson.scriptsdir("structs","forDisplayStructs.jl")
@everywhere include($dirToFileStructs)
@everywhere include($dirToImageStructs)

using Main.FileStructs
@everywhere  using Main.ForDisplayStructs
@everywhere  import Main.ForDisplayStructs

dirToH = DrWatson.scriptsdir("loadData","manageH5File.jl")
dirToWorkerPrepare = DrWatson.scriptsdir("mainPipeline","processesDefinitions","mainProcessDefinition.jl")


include(dirToWorkerPrepare)
include(dirToFileStructs)
include(dirToH)

using Main.h5manag
initializeWorker(persistenceWorker, ["Documenter", "HDF5","ColorTypes","Parameters"], [Main.FileStructs.filePathAndModuleName(dirToH,"h5manag"),
Main.FileStructs.filePathAndModuleName(DrWatson.scriptsdir("structs","forDisplayStructs.jl"),"ForDisplayStructs")] )

## image viewer 


dirToImageHelper = DrWatson.scriptsdir("display","imageViewerHelper.jl")
dirToImageColorManag= DrWatson.scriptsdir("display","manageColorSets.jl")

dirToImageDisplay = DrWatson.scriptsdir("display","mainDisplay.jl")
include(dirToImageHelper)
include(dirToImageColorManag)

include(dirToImageDisplay)
# @everywhere 1 using  Main.MyImgeViewer
# @everywhere 1 using Main.imageViewerHelper

# dirToImageViewer = DrWatson.scriptsdir("structs","forDisplayStructs.jl")

# @async initializeWorker(imageViewerHelperNumb, ["Documenter"],
#  [filePathAndModuleName(dirToImageHelper,"imageViewerHelper")
#  ,filePathAndModuleName(dirToImageStructs,"ForDisplayStructs")
#  ,filePathAndModuleName(dirToImageViewer,"MyImgeViewer")] )



exmpleH = @spawnat persistenceWorker Main.h5manag.getExample()
arrr= fetch(exmpleH)

minimumm = -1000

maximumm = 2000
imageDim = size(arrr)
maskF = @spawnat persistenceWorker Main.h5manag.getOrCreateMaskData(Int16, "liverOrganMask", "trainingScans/liver-orig005.mhd", imageDim, RGBA(0,0,255,0.4))
mask = fetch(maskF)

using Main.imageViewerHelper
using Main.MyImgeViewer

 

singleCtScanDisplay(arrr, [mask],minimumm, maximumm)

# using GLMakie
# arrr[1,1,:].= minimumm 
# arrr[2,1,:].= maximumm 

# arrr[1,:,1].= minimumm 
# arrr[2,:,1].= maximumm 

# arrr[:,:,1].= minimumm 
# arrr[:,:,2].= maximumm 


# scene, layout = GLMakie.layoutscene(resolution = (600, 400))
# ax1 = layout[1, 1] = GLMakie.Axis(scene, backgroundcolor = :transparent)
# GLMakie.heatmap!(ax1, arrr[90,:,:] ,colormap = Main.ManageColorSets.createMedicalImageColorScheme(200,-200,maximumm, minimumm )) 
# scene