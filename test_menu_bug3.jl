using GLMakie
fig = Figure()
g = fig[1,1] = GridLayout()

all_opts = String["", "Abdomen", "Bone", "Chest", "Liver", "Prostate", "Zebra"]
# limit is 3 instead of 25
MAX_DISPLAY = 3

options = Observable(String["", "Abdomen", "Bone"])
menu = Menu(g[1,1], options=options)

# User selects Abdomen (which is in the first 3)
menu.i_selected[] = 2
println("Selection before close: ", menu.selection[])
sel = menu.selection[]

# Menu closes -> restore options
menu.options[] = all_opts[1:MAX_DISPLAY]
# Makie automatically resets i_selected to 1

current_opts = menu.options[]
if string(sel) ∉ current_opts
    menu.options[] = vcat([string(sel)], current_opts)
    menu.i_selected[] = 1
end

println("Selection after close (Abdomen): ", menu.selection[])

# --- NOW FOR PROSTATE ---
menu.options[] = String["", "Prostate"]
menu.i_selected[] = 2
println("Selection before close: ", menu.selection[])
sel = menu.selection[]

# Menu closes -> restore options
menu.options[] = all_opts[1:MAX_DISPLAY]
# Makie automatically resets i_selected to 1

current_opts = menu.options[]
if string(sel) ∉ current_opts
    menu.options[] = vcat([string(sel)], current_opts)
    menu.i_selected[] = 1
end

println("Selection after close (Prostate): ", menu.selection[])
