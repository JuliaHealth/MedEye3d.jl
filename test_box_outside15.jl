using GLMakie
fig = Figure()
g = GridLayout(fig[1,1])
r1 = 1; r2 = 2
m1 = Menu(g[r1, 1], options=["A"])
m2 = Menu(g[r2, 1], options=["B"])
b = Box(g[r1:r2, 1], color=(:transparent, 0.0), strokecolor=:blue, strokewidth=2, alignmode=Outside(-15))

Makie.GridLayoutBase.update!(g)
println("Box bbox with Outside(-15): ", b.layoutobservables.computedbbox[].widths)
println("Box bbox origin: ", b.layoutobservables.computedbbox[].origin)
println("m1 bbox origin & height: ", m1.layoutobservables.computedbbox[].origin, " ", m1.layoutobservables.computedbbox[].widths)
println("m2 bbox origin & height: ", m2.layoutobservables.computedbbox[].origin, " ", m2.layoutobservables.computedbbox[].widths)

save("test_box_outside15.png", fig)
