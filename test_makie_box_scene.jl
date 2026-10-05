using GLMakie
fig = Figure(size=(800, 600))
main_layout = GridLayout(fig[1, 1])

login_grid = GridLayout(bbox = fig.scene.viewport)
login_bg = Box(fig.scene, login_grid[1, 1], color=(:black, 0.92))

login_inner = GridLayout(login_grid[1, 1], halign=:center, valign=:center)
lbl = Label(fig.scene, login_inner[1, 1], "HELLO", color=:red)

println("Success!")
