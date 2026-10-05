using GLMakie
fig = Figure(size=(800, 600))
main_layout = GridLayout(fig[1, 1])
# Put a large Box in main_layout to simulate UI content
Box(main_layout[1, 1], width=800, height=600)

center_grid = GridLayout(fig[1, 1])
Box(center_grid[1, 1], visible=false)
Box(center_grid[3, 3], visible=false)
login_inner = GridLayout(center_grid[2, 2])

colsize!(center_grid, 1, Auto())
colsize!(center_grid, 2, Fixed(310))
colsize!(center_grid, 3, Auto())
rowsize!(center_grid, 1, Auto())
rowsize!(center_grid, 2, Fixed(175))
rowsize!(center_grid, 3, Auto())

colsize!(login_inner, 1, Fixed(100))
rowsize!(login_inner, 1, Fixed(100))
Box(login_inner[1,1], color=:red)

println("inner bbox: ", login_inner.layoutobservables.computedbbox[])
