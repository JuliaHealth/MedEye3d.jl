using GLMakie
fig = Figure(size=(800, 600))
main_layout = GridLayout(fig[1, 1])
lbl = Label(main_layout[1, 1], "Background Label")
colsize!(main_layout, 1, Fixed(800))
rowsize!(main_layout, 1, Fixed(600))

login_bg = Box(fig.layout[1:end, 1:end], color=(:black, 0.5))

login_inner = GridLayout(fig.layout[1:end, 1:end], tellwidth=false, tellheight=false, halign=:center, valign=:center)
colsize!(login_inner, 1, Fixed(100))
rowsize!(login_inner, 1, Fixed(100))
Box(login_inner[1,1], color=:red)

println("inner bbox: ", login_inner.layoutobservables.computedbbox[])
