using Pkg
Pkg.add("PackageCompiler")
using PackageCompiler
println("PackageCompiler is ready: ", pkgversion(PackageCompiler))
