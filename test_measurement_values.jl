using MedEye3d
using MedEye3d.Measurements

measurements = [Measurements.SphereMeasurement(id=1, center_idx=(256.0f0, 256.0f0, 50.0f0), radius_mm=10.0f0, suv_mean=0.0f0, suv_max=0.0f0)]

struct FakeCalcDims
    imageTextureWidth::Int
    imageTextureHeight::Int
end

mutable struct FakePanelState
    currentDisplayedSlice::Int
    calcDimsStruct::FakeCalcDims
    spacingsValue::Vector{NTuple{3,Float64}}
end

state = [FakePanelState(50, FakeCalcDims(512, 512), [(1.0, 1.0, 1.0)])]
panel_id = 1

verts = Measurements.compute_measurement_vertices(measurements, state, panel_id)

println("Generated ", length(verts), " floats.")
println("First 6 vertices:")
for i in 1:6
    idx = (i-1)*6 + 1
    println("v", i, ": UV(", verts[idx], ", ", verts[idx+1], ") RGBA(", verts[idx+2], ",", verts[idx+3], ",", verts[idx+4], ",", verts[idx+5], ")")
end
