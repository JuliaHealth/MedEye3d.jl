using GLMakie
using Dates

fig1 = Figure(size=(400, 300), backgroundcolor=:blue)
Label(fig1[1,1], "Window 1", color=:white)

fig2 = Figure(size=(400, 300), backgroundcolor=:red)
Label(fig2[1,1], "Window 2", color=:white)

screen1 = GLMakie.Screen(fig1.scene)
display(screen1, fig1)

screen2 = GLMakie.Screen(fig2.scene)
display(screen2, fig2)

# save to check they rendered? We can't easily save "multiple windows" 
# but we can see if it throws errors.
println("Two screens created successfully")
