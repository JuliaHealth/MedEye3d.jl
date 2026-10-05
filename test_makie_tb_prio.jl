using GLMakie
fig = Figure(size=(800, 600))
tb = Textbox(fig[1, 1], placeholder="Type here")
on(events(fig.scene).mousebutton, priority=65) do event
    return Consume(true)
end

# Check if the tb blockscene has a mousebutton listener and its priority
listeners = events(tb.blockscene).mousebutton.listeners
for (prio, dict) in listeners
    println("Prio: ", prio)
end
