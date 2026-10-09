using GLMakie
fig = Figure(size = (400, 300))
g = fig[1, 1] = GridLayout()
ar1 = 1; ar2 = 2
m1 = Menu(g[ar1, 1], options=["A"])
m2 = Menu(g[ar2, 1], options=["B"])

b = Box(g[ar1:ar2, 1], color=(:transparent, 0.0), strokecolor=:red, strokewidth=2, tellheight=false, tellwidth=false, alignmode=Outside(5))

rowsize!(g, ar1, Auto())
rowsize!(g, ar2, Auto())

on(fig.scene.events.window_open) do _
    println("Box: ", b.layoutobservables.computedbbox[])
end
save("test_box3.png", fig)
