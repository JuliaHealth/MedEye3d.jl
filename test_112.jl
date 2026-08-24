using Pkg
Pkg.activate(temp=true)
Pkg.develop(PackageSpec(path="d:/ITKIOWrapper"))
using ITKIOWrapper
println("ITKIOWrapper loaded successfully in Julia 1.12.5!")
