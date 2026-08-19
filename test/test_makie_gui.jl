using Pkg
Pkg.activate(".")

using MedEye3d
using MedEye3d.LesionMetadataWindow
import Observables
using GLMakie

# Create a standalone metadata window to test sections
active_lesion = Observables.Observable("Lesion 1")
lesion_ids = Observables.Observable(["Lesion 1", "Lesion 2"])

# Use a dummy channel
dummy_ch = Channel{Any}(100)

println("Creating metadata window...")
win = LesionMetadataWindow.create_metadata_window(active_lesion, lesion_ids, dummy_ch)
println("Window created, saving screenshots...")

# Make figure taller to capture everything
resize!(win.fig, 920, 2400)
save("/mnt/big/project_ssd/project_ssd/MedEye3d.jl/data/makie_gui_full.png", win.fig)
println("Full screenshot saved")
