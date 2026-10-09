using GLMakie

fig = Figure(size=(800, 600))
Box(fig[1, 1], color=:red)

login_inner = GridLayout(fig[1, 1], halign=:center, valign=:center, width=310, height=175, tellwidth=false, tellheight=false)
login_inner_bg = Box(login_inner[1:end, 1:end], color=:blue)
Label(login_inner[1, 1:2], "MedEye3d Login", color=:white, fontsize=24, font=:bold, halign=:center)
Label(login_inner[2, 1], "Username:", color=:white, fontsize=14, halign=:right)
Textbox(login_inner[2, 2], placeholder="Enter username", fontsize=14, width=200)

rowsize!(login_inner, 1, Fixed(40))
rowsize!(login_inner, 2, Fixed(35))
colsize!(login_inner, 1, Fixed(100))
colsize!(login_inner, 2, Fixed(210))

save("layout_test2.png", fig)
