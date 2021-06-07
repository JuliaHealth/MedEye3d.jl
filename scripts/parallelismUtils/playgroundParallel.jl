



usingTest(ColorTypes)

#from http://5.9.10.113/65385311/how-to-load-modules-or-scripts-dynamically-at-runtime-with-julia-1-5

function foo(m::AbstractString)
    include(m * ".jl")
    Main.eval(Meta.parse("using Main.$m"))
    mod = getfield(Main, Symbol(m))
    Base.invokelatest(mod.hello)
end

Main.eval(Meta.parse("using ColorTypes"))




using DrWatson
@quickactivate "Probabilistic medical segmentation"
using Distributed
addprocs(1)

@everywhere using Pkg
@everywhere Pkg.instantiate()

using HDF5

dirToH = DrWatson.scriptsdir("loadData","manageH5File.jl")
dirToWorkerPrepare = DrWatson.scriptsdir("mainPipeline","processesDefinitions","mainProcessDefinition.jl")
dirToFileStructs = DrWatson.scriptsdir("structs","forFilesEtc.jl")
include(dirToFileStructs)



include(dirToWorkerPrepare)
include(dirToFileStructs)
include(dirToH)

using Main.h5manag
x = initializeWorker(2, ["Documenter", "HDF5","ColorTypes"], [filePathAndModuleName(dirToH,"h5manag")] )

fetch(x)


exmpleH = @spawnat 2 Main.h5manag.getExample()

fetch(exmpleH)