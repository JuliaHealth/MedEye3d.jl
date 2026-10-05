using GLMakie
fig = Figure()
lbl1 = Label(fig[1, 1], "Label 1")
lbl2 = Label(fig[5, 5], "Label 2")
g = GridLayout(fig.layout[1:end, 1:end])
println(g.layoutobservables.suggestedbbox[])
