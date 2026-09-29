using GLMakie
fig = Figure()
g = fig[1,1] = GridLayout()
options = Observable(String["", "Prostate", "Liver"])
menu = Menu(g[1,1], options=options)

menu.i_selected[] = 2 # Select Prostate
println("Selection: ", menu.selection[])

struct_sel = menu.selection[]
struct_str = struct_sel === nothing ? "" : strip(string(struct_sel))
println("Struct str: '", struct_str, "'")
