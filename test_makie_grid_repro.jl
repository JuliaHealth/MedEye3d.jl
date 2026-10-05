using GLMakie
fig = Figure(size=(800, 600))
main_layout = GridLayout(fig[1, 1])
lbl = Label(main_layout[1, 1], "Background Label")

_login_visible = Observable(true)

login_bg = Box(fig.layout[1:end, 1:end], color=(:black, 0.92), strokewidth=0, visible=_login_visible)

login_inner = GridLayout(fig.layout[1:end, 1:end], tellwidth=false, tellheight=false, halign=:center, valign=:center)
login_inner_bg = Button(login_inner[1:end, 1:end], label="", buttoncolor=:transparent, buttoncolor_active=:transparent, buttoncolor_hover=:transparent, strokewidth=0, width=nothing, height=nothing)

lbl_title = Label(login_inner[1, 1:2], "MedEye3d Login", color=:white, fontsize=24, font=:bold, halign=:center, visible=_login_visible)
lbl_user = Label(login_inner[2, 1], "Username:", color=:white, fontsize=14, halign=:right, visible=_login_visible)
login_tb_user = Textbox(login_inner[2, 2], placeholder="Enter username", fontsize=14, width=200)
lbl_pass = Label(login_inner[3, 1], "Password:", color=:white, fontsize=14, halign=:right, visible=_login_visible)
login_tb_pass = Textbox(login_inner[3, 2], placeholder="Enter password", fontsize=14, width=200)
login_btn = Button(login_inner[4, 1:2], label="Login")

rowsize!(login_inner, 1, Fixed(40))
rowsize!(login_inner, 2, Fixed(35))
rowsize!(login_inner, 3, Fixed(35))
rowsize!(login_inner, 4, Fixed(40))
colsize!(login_inner, 1, Fixed(100))
colsize!(login_inner, 2, Fixed(210))

println("inner bbox: ", login_inner.layoutobservables.computedbbox[])
