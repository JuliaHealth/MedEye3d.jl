using Pkg
Pkg.precompile()
using MedImages
println("MedImages loaded successfully! Version: ", pkgversion(MedImages))
using MedEye3d
println("MedEye3d loaded successfully! Version: ", pkgversion(MedEye3d))
