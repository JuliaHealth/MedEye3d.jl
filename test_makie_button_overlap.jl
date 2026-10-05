using GLMakie
fig = Figure(size=(800, 600))
login_inner = GridLayout(fig[1, 1], halign=:center, valign=:center)
btn = Button(login_inner[1:end, 1:end], label="BG", buttoncolor=:gray)
tb = Textbox(login_inner[1, 1], placeholder="Enter text", width=200)

# Simulate a click on the textbox!
# Actually we can check z-index or event priority!
println("tb blockscene priority: ", tb.blockscene)
