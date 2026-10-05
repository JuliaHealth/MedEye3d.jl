using GLMakie
include("src/MedEye3d.jl")
using .MedEye3d

# Create dummy state
import .MedEye3d.SegmentationDisplay.MakieEventHandlers: StateDataFields, CalcDimsStruct, BaseDatAndTexture
import .MedEye3d.SegmentationDisplay: MainForDisplayObjects

st = StateDataFields(
    Observable(nothing),
    CalcDimsStruct(Float32[], Float32[], Float32[], Float32[], Float32[]),
    nothing,
    MainForDisplayObjects(nothing, nothing, nothing, [], nothing, nothing),
    0, false, 0, 0.0f0, 0, 0
)

# Launch the window
fig, lmw_obs = MedEye3d.LesionMetadataWindow.create_metadata_window([st], nothing)
GLMakie.activate!()
screen = display(fig)

# Show login
lmw_obs[:login_visible][] = true

# Save screenshot
sleep(2)
save("login_screenshot.png", fig)
