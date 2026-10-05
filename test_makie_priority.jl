using GLMakie
fig = Figure(size=(800, 600))
btn = Button(fig[1, 1], label="CLICK ME")
on(btn.clicks) do _
    println("BUTTON CLICKED!")
end

on(events(fig.scene).mousebutton, priority=65) do event
    println("Priority 65 hit! Consuming...")
    return Consume(true)
end

# Simulate a click!
# I am not sure how to simulate a click, but I can check button priority.
println("Button blocks priority? ", btn.blockscene)
