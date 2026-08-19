using Makie
using GLMakie
fig = Figure()
menu = Menu(fig[1, 1], options = ["A", "B"], default = "A")
println("Initial selection: ", typeof(menu.selection[]), " value: ", menu.selection[])
