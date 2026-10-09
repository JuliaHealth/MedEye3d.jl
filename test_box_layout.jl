using GLMakie

fig = Figure(size = (400, 300))
g = fig[1, 1] = GridLayout()

Label(g[1, 1], "Row 1")
Label(g[2, 1], "Row 2")
Label(g[3, 1], "Row 3")

# test default Box spanning rows 1:2
b = Box(g[1:2, 1], color=(:transparent, 0.0), strokecolor=:red, strokewidth=2, tellheight=false, tellwidth=false)

rowsize!(g, 1, 30)
rowsize!(g, 2, 30)

# get coordinates
on(fig.scene.events.window_open) do _
    bb = b.layoutobservables.computedbbox[]
    println("Box BBox: ", bb)
end

save("test_box.png", fig)
