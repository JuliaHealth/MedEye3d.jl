using MedEye3d; using GLMakie
fig = Figure()
g = GridLayout(fig[1,1])
btn = Button(g[1,1], label="", width=nothing, height=nothing)
println(btn.layoutobservables.suggestedbbox[])
