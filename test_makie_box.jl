using GLMakie
fig = Figure(size=(800, 600))
login_grid = GridLayout(bbox = fig.layout.layoutobservables.suggestedbbox)
login_bg = Box(fig, login_grid[1, 1], color=:red)
println("Success!")
