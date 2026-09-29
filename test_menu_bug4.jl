using GLMakie
fig = Figure()
g = fig[1,1] = GridLayout()

all_opts = String["", "Abdomen", "Bone", "Chest", "Liver", "Prostate", "Zebra"]
MAX_DISPLAY = 3

options = Observable(all_opts[1:MAX_DISPLAY])
menu = Menu(g[1,1], options=options)

# User opens menu and types "P"
query = "P"
filtered = filter(x -> contains(lowercase(x), lowercase(query)), all_opts)
menu.options[] = filtered # ["Prostate"]
menu.i_selected[] = 1
println("Selection after filter and select: ", menu.selection[])
sel = menu.selection[]

# User closes menu
menu.options[] = all_opts[1:MAX_DISPLAY]
current_opts = menu.options[]
if string(sel) ∉ current_opts
    menu.options[] = vcat([string(sel)], current_opts)
    menu.i_selected[] = 1
end

println("Selection after close: ", menu.selection[])
