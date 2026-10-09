using GLMakie
using Observables

fig1 = Figure(size=(400, 300))
screen1 = GLMakie.Screen(fig1.scene)
display(screen1, fig1)

show_login = Observable(false)
on(show_login) do vis
    if vis
        fig2 = Figure(size=(400, 300), backgroundcolor=:blue)
        Label(fig2[1,1], "Login Window", color=:white)
        screen2 = GLMakie.Screen(fig2.scene)
        display(screen2, fig2)
        println("Created and displayed screen2")
    end
end

show_login[] = true
sleep(2)
