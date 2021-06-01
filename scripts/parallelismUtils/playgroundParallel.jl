

using Distributed

function include_remote(path, workers=workers(); mod=Main)
    open(path) do f
    text, s = read(f, String), 1
    while s <= length(text)
    ex, s = Meta.parse(text, s) # Parse text starting at pos s, return new s
    for w in workers
    @spawnat w Core.eval(mod, ex) # Evaluate the expression on workers
    end
    end
    end
    end
    

using Pkg
using DrWatson
@quickactivate "Probabilistic medical segmentation"
addprocs(2)

workers()[1]

@everywhere 2 using Pkg
@everywhere 2 pkg"activate ." # Activate the current directory 
@everywhere 2 pkg"instantiate"
@everywhere 2 pkg"precompile"
@everywhere 2 Pkg.add("ColorTypes")
@everywhere 2 using ColorTypes

@everywhere 2 Pkg.add("Documenter")
@everywhere 2 using Documenter

@everywhere 2 Pkg.add("HDF5")
@everywhere 2 using HDF5

using HDF5

dirToH = DrWatson.scriptsdir("loadData","manageH5File.jl")

# include(dirToH)
# using Main.h5manag


include_remote(dirToH)
@everywhere 2 using Main.h5manag

zz = @spawnat 2 keys(Main.h5manag.aaa())


# @everywhere 2 Main.h5manag.setG($gp)

exmpleH = @spawnat 2 Main.h5manag.getExample()

fetch(exmpleH)

