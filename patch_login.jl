content = read("src/display/LesionMetadataWindow.jl", String)

new_layout = """    login_bg = Box(fig[1, 1], color=(:black, 0.92), strokewidth=0, visible=_login_visible)
    
    login_inner = GridLayout(fig[1, 1], tellwidth=false, tellheight=false, halign=:center, valign=:center)
    login_inner_bg = Button(login_inner[1:end, 1:end], label="", buttoncolor=:transparent, buttoncolor_active=:transparent, buttoncolor_hover=:transparent, strokewidth=0, width=nothing, height=nothing)"""

content = replace(content, r"    login_bg = Box\(fig.layout\[1:end, 1:end\].*?login_inner_bg = Button\(login_inner\[1:end, 1:end\], label=\"\", buttoncolor=:transparent, buttoncolor_active=:transparent, buttoncolor_hover=:transparent, strokewidth=0, width=nothing, height=nothing\)"s => new_layout)

write("src/display/LesionMetadataWindow.jl", content)
println("Patched!")
