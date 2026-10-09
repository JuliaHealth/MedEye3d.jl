using GLMakie
fig = Figure(size = (400, 300))
g = fig[1, 1] = GridLayout()
ar1 = 1; ar2 = 2
m1 = Menu(g[ar1, 1], options=["A"])
m2 = Menu(g[ar2, 1], options=["B"])

b = Box(g[ar1:ar2, 1], color=(:transparent, 0.0), strokecolor=:red, strokewidth=2, tellheight=false, tellwidth=false)

rowsize!(g, ar1, Auto())
rowsize!(g, ar2, Auto())

on(fig.scene.events.window_open) do _
    bb_b = b.layoutobservables.computedbbox[]
    bb_m1 = m1.layoutobservables.computedbbox[]
    bb_m2 = m2.layoutobservables.computedbbox[]
    println("Box: ", bb_b)
    println("M1:  ", bb_m1)
    println("M2:  ", bb_m2)
end
save("test_box2.png", fig)
