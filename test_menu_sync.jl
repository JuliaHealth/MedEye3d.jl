using GLMakie
fig = Figure()
m = Menu(fig[1,1], options=["A", "B", "C"])
m.i_selected[] = 2
println(m.selection[])
