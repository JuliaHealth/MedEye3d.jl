using GLMakie
fig = Figure(size=(800, 600))
lbl = Label(fig[1, 1], "Background Label")

login_grid = GridLayout(fig.scene, bbox = fig.layout.layoutobservables.suggestedbbox)
login_bg = Box(login_grid[1, 1], color=(:black, 0.5))

login_inner = GridLayout(login_grid[1, 1], tellwidth=false, tellheight=false, halign=:center, valign=:center)
colsize!(login_inner, 1, Fixed(100))
rowsize!(login_inner, 1, Fixed(100))
Box(login_inner[1,1], color=:red)

println("inner bbox: ", login_inner.layoutobservables.computedbbox[])
