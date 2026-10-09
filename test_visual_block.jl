using GLMakie
fig = Figure()
g = GridLayout(fig[1,1])
m1 = Menu(g[1,1], options=["A"])
m2 = Menu(g[2,1], options=["B"])
g.block_updates = true

on(fig.scene.events.mousebutton) do event
    if event.action == Mouse.press
        rowsize!(g, 2, Fixed(0))
        m2.blockscene.visible[] = false
    end
    return Consume(false)
end

save("test_visual_block_before.png", fig)
rowsize!(g, 2, Fixed(0))
m2.blockscene.visible[] = false
save("test_visual_block_after.png", fig)
