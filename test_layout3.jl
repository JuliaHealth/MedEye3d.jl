using GLMakie

fig = Figure(size=(800, 600))
Box(fig[1, 1], color=:red) # background

center_grid = GridLayout(fig[1, 1]) # NO tellwidth=false!
Box(center_grid[1, 1], visible=false)
Box(center_grid[3, 3], visible=false)
colsize!(center_grid, 1, Auto())
colsize!(center_grid, 2, Fixed(310))
colsize!(center_grid, 3, Auto())
rowsize!(center_grid, 1, Auto())
rowsize!(center_grid, 2, Fixed(175))
rowsize!(center_grid, 3, Auto())

login_inner = GridLayout(center_grid[2, 2])
login_inner_bg = Box(login_inner[1:end, 1:end], color=:blue)
Label(login_inner[1, 1:2], "MedEye3d Login", color=:white, fontsize=24, font=:bold, halign=:center)
Label(login_inner[2, 1], "Username:", color=:white, fontsize=14, halign=:right)
Textbox(login_inner[2, 2], placeholder="Enter username", fontsize=14, width=200)

rowsize!(login_inner, 1, Fixed(40))
rowsize!(login_inner, 2, Fixed(35))
colsize!(login_inner, 1, Fixed(100))
colsize!(login_inner, 2, Fixed(210))

save("layout_test3.png", fig)
