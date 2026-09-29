using GLMakie
fig = Figure()
g = fig[1,1] = GridLayout()

all_opts = String["", "Abdomen", "Bone", "Chest", "Liver", "Prostate", "Zebra"]
MAX_DISPLAY = 3

options = Observable(all_opts[1:MAX_DISPLAY])
menu = Menu(g[1,1], options=options)

# User clicks to open the menu
menu.is_open[] = true

# User types "P"
query = "P"
filtered = filter(x -> contains(lowercase(x), lowercase(query)), all_opts)
menu.options[] = filtered # ["Prostate"]
# User clicks "Prostate"
menu.i_selected[] = 1
sel = menu.selection[]

# User clicks OUTSIDE to close the menu
menu.is_open[] = false

# searchable_menu closing logic:
menu.options[] = all_opts[1:MAX_DISPLAY] # Makie resets i_selected[] = 1 here!!
current_opts = menu.options[]
if string(sel) ∉ current_opts
    menu.options[] = vcat([string(sel)], current_opts)
    menu.i_selected[] = 1
end

println("Selection after close: ", menu.selection[])
