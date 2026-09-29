using MedEye3d; using GLMakie
fig = Figure()
m1 = Menu(fig[1,1], options=["A", "B"])
m2 = Menu(fig[2,1], options=["C", "D"])
m1.blockscene.visible[] = false
println("Menu 1 visible: ", m1.blockscene.visible[])
