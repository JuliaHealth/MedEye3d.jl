path = "src/display/LesionMetadataWindow.jl"
content = read(path, String)

bad_snippet = """
    # Loading Overlay for Makie
    overlay_grid = GridLayout(fig.layout[1:end, 1:end], tellwidth=false, tellheight=false, halign=:fill, valign=:fill)
    loading_bg = Box(overlay_grid[1, 1], color=(:black, 0.85), strokewidth=0)
    loading_txt = Label(overlay_grid[1, 1], "LOADING APPLICATION...\\nPlease wait while engines initialize.", color=:white, fontsize=30, font=:bold, halign=:center, valign=:center)
    
    _lmw_observables[:loading_overlay_bg] = loading_bg
    _lmw_observables[:loading_overlay_txt] = loading_txt

    return result
end"""

good_snippet = """
    return result
end"""

content = replace(content, bad_snippet => good_snippet)
write(path, content)
println("Cleaned up bad injections.")
