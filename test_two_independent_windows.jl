using GLMakie

f = x -> nothing

fig1 = Figure(size=(400, 300))
s1 = GLMakie.Screen(fig1.scene; renderloop=f)
display(s1, fig1)

fig2 = Figure(size=(350, 200))
# DO NOT pass fig2.scene
s2 = GLMakie.Screen(; renderloop=f)
display(s2, fig2)

println("s1.glscreen == s2.glscreen? ", s1.glscreen == s2.glscreen)

GLMakie.GLFW.HideWindow(s2.glscreen)
println("s2 hidden. Is s1 open? ", isopen(s1))
