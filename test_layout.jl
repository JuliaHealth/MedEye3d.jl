using GLMakie

fig = Figure(size=(800, 600))
# Add something to make fig[1, 1] large
Box(fig[1, 1], color=:red)

# Try halign on GridLayout
gl = GridLayout(fig[1, 1], halign=:center, valign=:center, width=310, height=175, tellwidth=false, tellheight=false)
Box(gl[1, 1], color=:blue)
Label(gl[1, 1], "Login Modal", color=:white)

save("layout_test.png", fig)
