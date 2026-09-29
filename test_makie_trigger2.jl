using GLMakie

fig = Figure()
options = Observable(String["A", "B"])
menu = Menu(fig[1,1], options=options)

println("Initial selection: ", menu.selection[])

on(menu.selection) do val
    println("Triggered! Value: ", val)
end

println("Changing options...")
options[] = String["C", "D"]
println("Final selection: ", menu.selection[])
