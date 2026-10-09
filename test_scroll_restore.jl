using GLMakie
fig = Figure()
g = GridLayout(fig[1,1])
m1 = Menu(g[1,1], options=["A"])
m2 = Menu(g[2,1], options=["B"])

# Initial layout
Makie.GridLayoutBase.update!(g)
orig_y = m2.layoutobservables.computedbbox[].origin[2]
println("Orig y: ", orig_y)

# Simulate scroll offset +100
offset = 100.0f0
m2.layoutobservables.computedbbox[] = Rect2f(m2.layoutobservables.computedbbox[].origin[1], orig_y + offset, m2.layoutobservables.computedbbox[].widths[1], m2.layoutobservables.computedbbox[].widths[2])
println("Scrolled y: ", m2.layoutobservables.computedbbox[].origin[2])

# Now update!(g)
Makie.GridLayoutBase.update!(g)
println("After update!(g) y: ", m2.layoutobservables.computedbbox[].origin[2])

