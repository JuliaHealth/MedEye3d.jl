using GLMakie
fig = Figure(size=(800, 600))
btn = Button(fig[1, 1], label="CLICK")
listeners = events(btn.blockscene).mousebutton.listeners
for (prio, dict) in listeners
    println("Btn Prio: ", prio)
end
