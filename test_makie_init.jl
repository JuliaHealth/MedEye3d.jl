using GLMakie

fig = Figure()
options = Observable(String["A", "B"])
menu = Menu(fig[1,1], options=options)

println("Initial selection: ", menu.selection[])
