using GLMakie
fig = Figure(size=(800, 600))
m = Menu(fig[1, 1], options=["A", "B"])
listeners = events(m.blockscene).mousebutton.listeners
for (prio, dict) in listeners
    println("Menu Prio: ", prio)
end
