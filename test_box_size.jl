using GLMakie
fig = Figure()
g = GridLayout(fig[1,1])
rowgap!(g, 10)
r1=1; r2=2
m1 = Menu(g[r1,1], options=["A"], height=30)
m2 = Menu(g[r2,1], options=["B"], height=30)
b = Box(g[r1:r2, 1], color=(:red,0.2), alignmode=Outside(-10))
save("test_box_size.png", fig)
