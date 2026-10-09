using GLMakie

fig1 = Figure()
s1 = GLMakie.Screen(fig1.scene)

fig2 = Figure()
s2 = GLMakie.Screen(fig2.scene; renderloop=nothing)

s3 = GLMakie.Screen()

println("s1 == s2? ", s1.glscreen == s2.glscreen)
println("s1 == s3? ", s1.glscreen == s3.glscreen)
