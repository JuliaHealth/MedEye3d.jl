using GLMakie

fig1 = Figure(size=(400, 300), backgroundcolor=:red)
screen1 = GLMakie.Screen() # global singleton
display(screen1, fig1)

fig2 = Figure(size=(400, 300), backgroundcolor=:blue)
screen2 = GLMakie.Screen(fig2.scene; visible=true) # does this force a new one?
display(screen2, fig2)

println("screen1.glscreen == screen2.glscreen? ", screen1.glscreen == screen2.glscreen)
