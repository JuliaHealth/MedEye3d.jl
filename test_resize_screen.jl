using GLMakie

f = x -> nothing

fig2 = Figure(size=(350, 200))
s2 = GLMakie.Screen(; renderloop=f)
display(s2, fig2)

# Get the size of s2's GLFW window
w, h = GLMakie.GLFW.GetWindowSize(s2.glscreen)
println("Window size: ", w, "x", h)
