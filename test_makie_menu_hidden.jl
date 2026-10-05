using GLMakie
fig = Figure(size=(800, 600))
m = Menu(fig[1, 1], options=["A", "B"])
m.blockscene.visible[] = false

tb = Textbox(fig[1, 1], placeholder="Type here")
# Does the menu steal the click from the textbox?
# I can verify by checking if clicking the textbox focuses it.
# Actually, I can just print the event consumption.
