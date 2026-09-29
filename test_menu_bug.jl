using GLMakie
fig = Figure()
g = fig[1,1] = GridLayout()
options = Observable(String["", "Prostate", "Liver"])
all_opts = String["", "Prostate", "Liver"]

menu = Menu(g[1,1], options=options)

menu.i_selected[] = 2 # Select Prostate
println("Selection before option change: ", menu.selection[])

# Now simulate closing the menu which restores all_opts
menu.options[] = all_opts
println("Selection after option change: ", menu.selection[])
