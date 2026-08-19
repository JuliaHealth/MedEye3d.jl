using GLMakie
fig = Figure()
m = Menu(fig[1,1], options=["A", "B"], default="A")
println("Selection is: ", repr(m.selection[]))
