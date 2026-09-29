using MedEye3d; using GLMakie
fig = Figure()
m = Menu(fig[1,1], options=["A", "B"])
println(hasproperty(m, :blockscene))
println(hasproperty(m.blockscene, :visible))
