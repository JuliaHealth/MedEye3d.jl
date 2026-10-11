using GLMakie
fig = Figure()
b = Button(fig[1, 1], label = "Hover me")
t = Tooltip(fig, [0,0], "Tooltip text")
display(fig)
