using GLMakie
fig = Figure()
m = Menu(fig[1,1], options=["A"])
println(propertynames(m))
