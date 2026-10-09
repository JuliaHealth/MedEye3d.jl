using GLMakie
fig = Figure()
g = GridLayout(fig[1,1])
rowgap!(g, 10)
r1=1; r2=2
m1 = Menu(g[r1,1], options=["A"])
m2 = Menu(g[r2,1], options=["B"])
b = Box(g[r1:r2, 1], color=(:red,0.2), alignmode=Outside(-15))
save("test_box_size2.png", fig)
