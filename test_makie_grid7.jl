using GLMakie
fig = Figure(size=(800, 600))
main_layout = GridLayout(fig[1, 1])
lbl = Label(main_layout[1, 1], "Background Label", tellwidth=false, tellheight=false)
colsize!(main_layout, 1, Fixed(800))
rowsize!(main_layout, 1, Fixed(600))

login_grid = GridLayout(fig[1, 1])
login_bg = Box(login_grid[1, 1], color=(:black, 0.5))

login_inner = GridLayout(login_grid[1, 1], tellwidth=false, tellheight=false, halign=:center, valign=:center)
colsize!(login_inner, 1, Fixed(100))
rowsize!(login_inner, 1, Fixed(100))
Box(login_inner[1,1], color=:red)

println("inner bbox: ", login_inner.layoutobservables.computedbbox[])
