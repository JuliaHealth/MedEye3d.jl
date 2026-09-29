using GLMakie
fig = Figure()
m = Menu(fig[1,1], options=["A", "B"])
child_scene = Scene(fig.scene, clear=false, camera=campixel!)
btn = Button(child_scene, label="Login")
# To position btn, we can just let it sit.
on(btn.clicks) do _
    println("Button clicked!")
end
on(m.is_open) do _
    println("Menu opened!")
end
println("Ready.")
