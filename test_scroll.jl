using GLMakie
fig = Figure(size=(400, 800))
# Create a subscene
subscene = Scene(fig.scene, camera=campixel!)
g = GridLayout(subscene)
for i in 1:20
    Label(g[i, 1], "Row $i")
    Button(g[i, 2], label="Btn $i")
end
# Try adding a slider
sl = Slider(fig[1, 2], range=0:100, horizontal=false)
on(sl.value) do v
    translate!(subscene, 0, v, 0)
end
display(fig)
println("Done")
