using GLMakie
fig = Figure(resolution = (800, 600))
g = GridLayout(fig[1,1], halign=:fill)
println(typeof(g))
