using GLMakie
fig = Figure()
g = GridLayout(fig[1,1])
m1 = Menu(g[1,1], options=["A"])
m2 = Menu(g[2,1], options=["B"])

Makie.GridLayoutBase.update!(g)
orig_y = m2.layoutobservables.computedbbox[].origin[2]

# Scroll down by 150px
scroll_offset = 150.0f0
m2.layoutobservables.computedbbox[] = Rect2f(m2.layoutobservables.computedbbox[].origin[1], orig_y + scroll_offset, m2.layoutobservables.computedbbox[].widths[1], m2.layoutobservables.computedbbox[].widths[2])
println("Scrolled y: ", m2.layoutobservables.computedbbox[].origin[2])

# Now simulate update_anatomy_ui with scroll restoration
Makie.GridLayoutBase.update!(g)
# Re-cache and restore scroll
new_orig_y = m2.layoutobservables.computedbbox[].origin[2]
m2.layoutobservables.computedbbox[] = Rect2f(m2.layoutobservables.computedbbox[].origin[1], new_orig_y + scroll_offset, m2.layoutobservables.computedbbox[].widths[1], m2.layoutobservables.computedbbox[].widths[2])

println("After update & restore y: ", m2.layoutobservables.computedbbox[].origin[2])

