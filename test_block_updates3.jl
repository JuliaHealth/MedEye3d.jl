using GLMakie
fig = Figure()
g = GridLayout(fig[1,1])
b = Box(g[1,1])
g.block_updates = true
rowsize!(g, 1, 100)
println("with block_updates=true: ", b.layoutobservables.computedbbox[].widths[2])

g.block_updates = false
rowsize!(g, 1, 200)
println("with block_updates=false: ", b.layoutobservables.computedbbox[].widths[2])
