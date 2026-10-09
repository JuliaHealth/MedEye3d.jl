using GLMakie

fig = Figure(size=(400, 300))
screen = GLMakie.Screen(fig.scene)
display(screen, fig)
close(screen)
println("Closed!")
