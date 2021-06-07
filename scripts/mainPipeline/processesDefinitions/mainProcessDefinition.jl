using DrWatson
@quickactivate "Probabilistic medical segmentation"

dirToFileStructs = DrWatson.scriptsdir("structs","forFilesEtc.jl")
include(dirToFileStructs)

#dirToFileStructs = DrWatson.scriptsdir("structs","forFilesEtc.jl")
#include(dirToFileStructs)
using Distributed
using Pkg
#using Main.fileStructs
```@doc
given workerid list of packages, list of files to include with module names
it will initialize all of it to make worker process ready for worker
    workerNumbId - id of a worker we want to initialize
    Packages - list of names of packages we need to add and used
    filesToinclude - files that was written in this module and we want to include plus modules that are there
```
function initializeWorker(workerNumbId::Int, Packages::Vector{String}, filesToinclude )
    intitialActivation(workerNumbId) # initial activation
    for x in Packages 
        addPackage(x,workerNumbId ) 
    end
    for x in filesToinclude
        include_remote(x.filePath, x.moduleName,workerNumbId)
    end
    return true
end # function initializeWorker()   

```@doc
initial activation of a worker  process of given id - workerNumbId needed before next acctions will be done
```
function intitialActivation(workerNumbId::Int)
    @spawnat workerNumbId Main.eval(Meta.parse("using Pkg"))
    @spawnat workerNumbId pkg"activate ." # Activate the current directory 
    #@spawnat workerNumbId pkg"instantiate"
    #@spawnat workerNumbId pkg"precompile"
end # function intitialActivation


```@doc
adding and activating package required for this process to work
```
function addPackage(packageName::String,workerNumbId::Int)
    @spawnat workerNumbId Pkg.add(packageName)
    @spawnat workerNumbId Main.eval(Meta.parse("using $packageName"))
end# function addPackages()


```@doc
Basically this function is defined for adding files written in this project
given path to the julia file and worker number this function will include the file under this path
to the given worker we need also to supply the name of the module that is in the included file in order to make it usable ..
```
function include_remote(path::String,  moduleName::String,workerNumbId::Int; mod=Main)
    open(path) do f
    text, s = read(f, String), 1
    while s <= length(text)
    ex, s = Meta.parse(text, s) # Parse text starting at pos s, return new s
    @spawnat workerNumbId Core.eval(mod, ex) # Evaluate the expression on workers
    @spawnat workerNumbId Main.eval(Meta.parse("using Main.$moduleName"))
    end
    end
    end

