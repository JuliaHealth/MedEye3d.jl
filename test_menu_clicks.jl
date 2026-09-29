using GLMakie
fig = Figure()
m = Menu(fig[1,1], options=["A", "B"])
m.blockscene.visible[] = false

on(m.is_open) do open
    println("Menu opened: ", open)
end
println("Simulating click...")
# How to simulate a click?
events(fig.scene).mousebutton[] = MouseButtonEvent(Mouse.left, Mouse.press)
events(fig.scene).mousebutton[] = MouseButtonEvent(Mouse.left, Mouse.release)
sleep(1)
