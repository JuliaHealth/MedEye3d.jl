using GLMakie
fig = Figure(size=(800, 600))
on(events(fig.scene).mousebutton) do event
    if event.action == Mouse.press
        println("pos: ", events(fig.scene).mouseposition[])
    end
end
