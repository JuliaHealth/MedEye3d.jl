using GLMakie
fig = Figure()
g = fig[1,1] = GridLayout()
options = Observable(String["", "A", "B", "C"])
all_opts = String["", "A", "B", "C"]

menu = Menu(g[1,1], options=options)

menu.selection[] = "Prostate" # Manually set to something not in options
println("Selection before option change: ", menu.selection[])

# Now simulate closing the menu
current_opts = menu.options[]
sel = menu.selection[]
if string(sel) ∉ current_opts
    menu.options[] = vcat([string(sel)], current_opts)
    menu.i_selected[] = 1
end
println("Selection after option change: ", menu.selection[])
