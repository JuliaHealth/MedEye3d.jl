# Run the production script but with auto-exit (no readline)
# Inject a screenshot command after everything initializes

# First, execute the production script
include("/workspaces/MedEye3d.jl/scripts/app/run_interactive_mrb.jl")

# If we get here, everything initialized successfully
println("=== PRODUCTION SCRIPT INITIALIZED SUCCESSFULLY ===")
println("Taking quad view screenshot...")

using MedEye3d.MakieEvents
ch = mainMedEye3dInstance.channel

done = Channel{Bool}(1)
put!(ch, ScreenshotEvent("/workspaces/MedEye3d.jl/data/scr/quadview_production.png", done))
take!(done)
println("Quad view screenshot saved!")
println("=== ALL DONE ===")
