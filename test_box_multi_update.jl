using GLMakie
fig = Figure()
g = GridLayout(fig[1,1])
r1 = 1; r2 = 2
m1 = Menu(g[r1, 1], options=["A"])
m2 = Menu(g[r2, 1], options=["B"])
b = Box(g[r1:r2, 1], color=(:transparent, 0.0), strokecolor=:blue, strokewidth=2)

# Start with row 2 hidden
rowsize!(g, r2, Fixed(0))
m2.blockscene.visible[] = false
b.visible[] = false

println("Initial box bbox: ", b.layoutobservables.computedbbox[].widths)

# Now show row 2
rowsize!(g, r2, Auto())
m2.blockscene.visible[] = true
b.visible[] = true

println("After rowsize Auto box bbox: ", b.layoutobservables.computedbbox[].widths)

# Try update!(g)
Makie.GridLayoutBase.update!(g)
println("After update!(g) box bbox: ", b.layoutobservables.computedbbox[].widths)

