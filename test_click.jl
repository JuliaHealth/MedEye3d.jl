using MedEye3d
using MedEye3d.SegmentationDisplay
using MedEye3d.SegmentationDisplay.MakieEventHandlers

# We don't need to load the GUI. We can just create the queue and the state objects, and call reactToAddAutoPet!

# Dummy structs
struct DummyTexture
    name::String
    isEditable::Bool
end

# We need a proper StateDataFields struct. It's too complex to mock.
