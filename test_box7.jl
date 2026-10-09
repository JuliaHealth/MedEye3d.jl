using GLMakie

fig = Figure()
g = GridLayout(fig[1, 1])
rowgap!(g, 2)
r1 = 1
r2 = 2

Menu(g[r1, 1], options=["A"])
Menu(g[r1, 2], options=["C"])
m2 = Menu(g[r2, 1], options=["B"])
Menu(g[r2, 2], options=["D"])

b = Box(g[r1:r2, 1:2], color=(:red, 0.2), alignmode=Outside(0))

save("test_box7.png", fig)
