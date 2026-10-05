using GLMakie
fig = Figure(size=(800, 600))
tb = Textbox(fig[1, 1], placeholder="Type here")
on(events(fig.scene).mousebutton, priority=65) do event
    println("65 hit")
    return Consume(true)
end
println("Ready")
