using GLMakie

fig1 = Figure()
screen1 = GLMakie.Screen(fig1.scene)
display(screen1, fig1)

fig2 = Figure()
screen2 = GLMakie.Screen(fig2.scene)
display(screen2, fig2)

println("screen1.glscreen == screen2.glscreen? ", screen1.glscreen == screen2.glscreen)
