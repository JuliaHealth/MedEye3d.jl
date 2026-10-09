using GLMakie

fig = Figure(size=(920, 900), figure_padding=0)
main_layout = GridLayout(fig[1,1])
Box(main_layout[1,1], color=:green)
rowsize!(main_layout, 1, Auto())
colsize!(main_layout, 1, Auto())

login_bg = Box(fig[1, 1], color=(:black, 0.92), strokewidth=0)
login_inner = GridLayout(fig[1, 1], halign=:center, valign=:center, width=310, height=175, tellwidth=false, tellheight=false)
login_inner_bg = Box(login_inner[1:end, 1:end], color=:blue)
Label(login_inner[1, 1:2], "MedEye3d Login", color=:white, fontsize=24, font=:bold, halign=:center)

rowsize!(login_inner, 1, Fixed(40))
rowsize!(login_inner, 2, Fixed(35))
colsize!(login_inner, 1, Fixed(100))
colsize!(login_inner, 2, Fixed(210))

save("layout_test6.png", fig)
