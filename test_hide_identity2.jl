using GLMakie

fig1 = Figure()
screen1 = GLMakie.Screen(fig1.scene)
display(screen1, fig1)

fig2 = Figure()
screen2 = GLMakie.Screen(fig2.scene)
display(screen2, fig2)

# Count how many GLFW windows exist
# In GLFW, we can't easily list all windows, but we can check if it creates two windows visually.
