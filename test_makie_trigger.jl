using GLMakie

fig = Figure()
options = Observable(String["A", "B"])
menu = Menu(fig[1,1], options=options)

triggered = false
on(menu.selection) do val
    global triggered = true
    println("Triggered! Value: ", val)
end

println("Changing options...")
options[] = String["C", "D"]
println("Triggered? ", triggered)
