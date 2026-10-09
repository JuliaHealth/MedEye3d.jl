using GLMakie

fig = Figure()
g = GridLayout(fig[1, 1])
rowgap!(g, 2)
r1 = 1
r2 = 2

Menu(g[r1, 1], options=["A"])
Menu(g[r2, 1], options=["B"])

b = Box(g[r1:r2, 1], color=(:red, 0.2), tellheight=false, tellwidth=false, alignmode=Outside(0))

save("test_box4.png", fig)
