path = "src/display/LesionMetadataWindow.jl"
content = read(path, String)
content = replace(content, r"    # Loading Overlay for Makie\n    overlay_grid = GridLayout\(fig\.layout\[1:end, 1:end\], tellwidth=false, tellheight=false, halign=:fill, valign=:fill\)\n    loading_bg = Box\(overlay_grid\[1, 1\], color=\(:black, 0\.85\), strokewidth=0\)\n    loading_txt = Label\(overlay_grid\[1, 1\], \"LOADING APPLICATION\.\.\.\\\nPlease wait while engines initialize\.\", color=:white, fontsize=30, font=:bold, halign=:center, valign=:center\)\n    \n    _lmw_observables\[:loading_overlay_bg\] = loading_bg\n    _lmw_observables\[:loading_overlay_txt\] = loading_txt\n\n    return res\nult\nend" => "    return result\nend")

content = replace(content, r"    # Loading Overlay for Makie\n    overlay_grid = GridLayout\(fig\.layout\[1:end, 1:end\], tellwidth=false, tellheight=false, halign=:fill, valign=:fill\)\n    loading_bg = Box\(overlay_grid\[1, 1\], color=\(:black, 0\.85\), strokewidth=0\)\n    loading_txt = Label\(overlay_grid\[1, 1\], \"LOADING APPLICATION\.\.\.\\\nPlease wait while engines initialize\.\", color=:white, fontsize=30, font=:bold, halign=:center, valign=:center\)\n    \n    _lmw_observables\[:loading_overlay_bg\] = loading_bg\n    _lmw_observables\[:loading_overlay_txt\] = loading_txt\n\n    return result\nend" => "    return result\nend")

write(path, content)
println("Wiped bad occurrences.")
