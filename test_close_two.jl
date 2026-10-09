using GLMakie
using Observables

fig1 = Figure(size=(400, 300), backgroundcolor=:red)
Label(fig1[1,1], "Main Window")
screen1 = GLMakie.Screen(fig1.scene)
display(screen1, fig1)

fig2 = Figure(size=(400, 300), backgroundcolor=:blue)
Label(fig2[1,1], "Login Window")
screen2 = GLMakie.Screen(fig2.scene)
display(screen2, fig2)

println("Both open. Closing screen2...")
close(screen2)

println("Is screen1 still open? ", isopen(screen1))
