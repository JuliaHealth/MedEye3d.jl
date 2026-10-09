using GLMakie

fig1 = Figure()
s1 = GLMakie.Screen() # This creates a new screen? Wait.
display(s1, fig1)

s2 = GLMakie.Screen() # This creates another new screen?
fig2 = Figure()
display(s2, fig2)

println("s1 == s2? ", s1.glscreen == s2.glscreen)
