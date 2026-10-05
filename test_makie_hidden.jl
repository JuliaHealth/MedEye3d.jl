using GLMakie
fig = Figure(size=(800, 600))
main = GridLayout(fig[1,1])
colsize!(main, 1, Fixed(800))
rowsize!(main, 1, Fixed(600))

inner = GridLayout(fig.layout[1:end, 1:end], halign=:center, valign=:center)
lbl = Label(inner[1, 1], "TEST", visible=false)
Box(inner[1, 1], color=:red, visible=false)

println("inner bbox hidden: ", inner.layoutobservables.computedbbox[])

lbl.visible = true
println("inner bbox visible: ", inner.layoutobservables.computedbbox[])
