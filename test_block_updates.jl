using GLMakie
fig = Figure()
g = GridLayout(fig[1,1])
m = Menu(g[1,1], options=["A"])
g.block_updates = true
rowsize!(g, 1, 100)
println(m.layoutobservables.computedbbox[].widths[2])
