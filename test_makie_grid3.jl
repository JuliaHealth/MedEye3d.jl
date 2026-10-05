using GLMakie
fig = Figure(size=(800, 600))
g = GridLayout(fig[1,1], halign=:center, valign=:center, tellwidth=false, tellheight=false)
colsize!(g, 1, Fixed(100))
rowsize!(g, 1, Fixed(100))
Box(g[1,1], color=:red)
display(fig)
println(g.layoutobservables.suggestedbbox[])
