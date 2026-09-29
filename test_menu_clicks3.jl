using GLMakie
fig = Figure()
m = Menu(fig[1,1], options=["A", "B"])
m.blockscene.visible[] = false

on(m.is_open) do open
    println("Menu opened: ", open)
end
println("Clicking...")
events(fig.scene).mouseposition[] = Tuple(m.layoutobservables.computedbbox[].origin) .+ (5, 5)
events(fig.scene).mousebutton[] = Makie.MouseButtonEvent(Makie.Mouse.left, Makie.Mouse.press)
events(fig.scene).mousebutton[] = Makie.MouseButtonEvent(Makie.Mouse.left, Makie.Mouse.release)
