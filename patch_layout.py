import re
with open("src/display/LesionMetadataWindow.jl", "r") as f:
    content = f.read()

new_layout = """    login_bg = Box(fig[1, 1], color=(:black, 0.92), strokewidth=0, visible=_login_visible)
    
    # Bulletproof layout with explicit Auto margins to center the 310x175 modal
    center_grid = GridLayout(fig[1, 1], tellwidth=false, tellheight=false)
    Box(center_grid[1, 1], visible=false)
    Box(center_grid[3, 3], visible=false)
    colsize!(center_grid, 1, Auto())
    colsize!(center_grid, 2, Fixed(310))
    colsize!(center_grid, 3, Auto())
    rowsize!(center_grid, 1, Auto())
    rowsize!(center_grid, 2, Fixed(175))
    rowsize!(center_grid, 3, Auto())

    login_inner = GridLayout(center_grid[2, 2])
    login_inner_bg = Button(login_inner[1:end, 1:end], label="", buttoncolor=:transparent, buttoncolor_active=:transparent, buttoncolor_hover=:transparent, strokewidth=0, width=nothing, height=nothing)"""

pattern = re.compile(r"    login_bg = Box\(fig\[1, 1\].*?login_inner_bg = Button\(login_inner\[1:end, 1:end\], label=\"\", buttoncolor=:transparent, buttoncolor_active=:transparent, buttoncolor_hover=:transparent, strokewidth=0, width=nothing, height=nothing\)", re.DOTALL)
content = pattern.sub(new_layout, content)

with open("src/display/LesionMetadataWindow.jl", "w") as f:
    f.write(content)
print("Patched successfully!")
