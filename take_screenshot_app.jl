using MedEye3d
using GLFW
using GLMakie
using Colors
using FileIO

# wait for the app to be active
sleep(2)

# get the global texture specs
texs = MedEye3d.AppMain.MEH.global_texture_specs
println("Panels: ", length(texs))

# Save the screenshot of the GLFW window
# Unfortunately GLFW doesn't easily give us the pixels without OpenGL.
# We can just use xwd!
