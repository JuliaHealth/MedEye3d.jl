using GLMakie
fig = Figure(size=(800, 600))
btn_under = Button(fig[1, 1], label="UNDER")
on(btn_under.clicks) do _
    println("UNDER CLICKED")
end

bg = Button(fig[1, 1], label="", buttoncolor=(:black, 0.5), strokewidth=0)
on(bg.clicks) do _
    println("BG CLICKED")
end

