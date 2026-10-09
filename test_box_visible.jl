using GLMakie
fig = Figure()
g = GridLayout(fig[1,1])
r1=1; r2=2
Menu(g[r1,1], options=["A"])
m2 = Menu(g[r2,1], options=["B"])

b = Box(g[r1:r2, 1], color=(:red,0.2), alignmode=Outside(6))
b.visible[] = false

rowsize!(g, r2, Fixed(0))
m2.blockscene.visible[] = false

# simulate event
on(fig.scene.events.mousebutton) do event
    if event.action == Mouse.press
        rowsize!(g, r2, Auto())
        m2.blockscene.visible[] = true
        b.visible[] = true
    end
    return Consume(false)
end

save("test_box_visible_initial.png", fig)
