using MedEye3d.MakieEvents
using Serialization

# Create the event
event = CompareTimePointsEvent(true)

# Serialize it so the main app can read it if we had a pipe,
# but wait! We can't easily inject it into the running app's channel!
