path = "src/packaging/AppMain.jl"
content = read(path, String)
content = replace(content, "obs = getfield(makie_win, :_lmw_observables)" => "obs = MedEye3d.LesionMetadataWindow._lmw_observables")
write(path, content)
println("Fixed _lmw_observables access")
