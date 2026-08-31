using Pkg
Pkg.activate("/workspaces/MedEye3d.jl")

using MedEye3d
using MedEye3d.ForDisplayStructs
using MedEye3d.DataStructs
using MedEye3d.MakieEvents
using ColorTypes
using GLFW
using NIfTI

println("=== COMPREHENSIVE FEATURE TEST ===")

image_path = "/root/data_decathlon/Task09_Spleen/imagesTr/spleen_10.nii.gz"
mask_path = "/root/data_decathlon/Task09_Spleen/labelsTr/spleen_10.nii.gz"
nii_img = niread(image_path)
nii_mask = niread(mask_path)
vol_img = Float32.(nii_img.raw)
vol_mask = Int16.(round.(nii_mask.raw))

# Create multi-discrete mask with colors (like production)
colors_mapped = [RGB(1.0, 0.0, 0.0), RGB(0.0, 1.0, 0.0), RGB(0.0, 0.0, 1.0)]

ts_img = TextureSpec{Float32}(name="CT", isMainImage=true, color=RGB(1.,1.,1.), minAndMaxValue=Float32.([-150,250]))
ts_mask = TextureSpec{Int16}(
    name="Mask", isMainImage=false,
    isMultiDiscreteMask=true, isIntegerTexture=true,
    colorSet=colors_mapped,
    maskContribution=Float32(0.5),
    minAndMaxValue=Int16.([0, length(colors_mapped)])
)
ts_bone = TextureSpec{Int8}(
    name="Bone_Overlay", isMainImage=false, isIntegerTexture=true,
    color=RGB(0.0, 1.0, 1.0), 
    minAndMaxValue=Int8.([0, 3]),
    maskContribution=Float32(0.5),
    isVisible=true
)

tsa = Vector{Vector{TextureSpec}}([TextureSpec[deepcopy(ts_img), deepcopy(ts_mask), deepcopy(ts_bone)]])

spacing = (1.,1.,1.); origin = (0.,0.,0.)
dts = Vector{DataToScrollDims}()
push!(dts, DataToScrollDims(imageSize=size(vol_img), voxelSize=spacing, dimensionToScroll=3))

inst = MedEye3d.SegmentationDisplay.coordinateDisplay(
    tsa, Float32(1.0), dts, [spacing], [origin],
    Dict{String,Vector}("supervoxel_vertices"=>[],"supervoxel_indices"=>[]),
    Dict{Int64,Dict{Int64,Dict{String,Any}}}()
)

# Create bone overlay (surface=1, marrow=2, both=3) from CT thresholding
bone_overlay = zeros(Int8, size(vol_img))
bone_region = vol_img .> 200
for z in 2:size(vol_img,3)-1, y in 2:size(vol_img,2)-1, x in 2:size(vol_img,1)-1
    if bone_region[x,y,z]
        is_edge = !bone_region[x-1,y,z] || !bone_region[x+1,y,z] || 
                  !bone_region[x,y-1,z] || !bone_region[x,y+1,z] ||
                  !bone_region[x,y,z-1] || !bone_region[x,y,z+1]
        if is_edge
            bone_overlay[x,y,z] = Int8(1)
        else
            bone_overlay[x,y,z] = Int8(2)
        end
    end
end
println("Bone overlay non-zero voxels: $(sum(bone_overlay .> 0))")

SD = FullScrollableDat(dataToScroll=[
    ThreeDimRawDat{Float32}(type=Float32, name="CT", dat=vol_img),
    ThreeDimRawDat{Int16}(type=Int16, name="Mask", dat=vol_mask),
    ThreeDimRawDat{Int8}(type=Int8, name="Bone_Overlay", dat=bone_overlay)
], dimensionToScroll=3)
MedEye3d.SegmentationDisplay.passDataForScrolling(inst, SD)
sleep(2)

ch = inst.channel
state = inst.states[1]

function capture_synced(ch, path)
    done = Channel{Bool}(1)
    put!(ch, ScreenshotEvent(path, done))
    take!(done)
end

# ============== TEST 1: Mask with semi-transparency ===============
println("\n--- TEST 1: Semi-transparent multi-discrete mask ---")
put!(ch, 30); sleep(3)
println("Slice: $(state.currentDisplayedSlice)")
capture_synced(ch, "/workspaces/MedEye3d.jl/feat_t1_mask.png")
println("Captured t1")

# ============== TEST 2: Zoom ===============
println("\n--- TEST 2: Zoom 1.6x ---")
for i in 1:5; put!(ch, ScrollZoomEvent(2.0)); sleep(0.2); end
sleep(0.5); put!(ch, 0); sleep(1)
println("Zoom: $(state.calcDimsStruct.zoom)")
capture_synced(ch, "/workspaces/MedEye3d.jl/feat_t2_zoom.png")
println("Captured t2")

# Reset zoom
state.calcDimsStruct.zoom = 1.0f0
state.calcDimsStruct.panX = 0.0f0
state.calcDimsStruct.panY = 0.0f0

# ============== TEST 3: Pan ===============
println("\n--- TEST 3: Pan ---")
state.calcDimsStruct.panX = 0.2f0
state.calcDimsStruct.panY = -0.15f0
state.calcDimsStruct.zoom = 1.5f0
put!(ch, 0); sleep(1)
capture_synced(ch, "/workspaces/MedEye3d.jl/feat_t3_pan.png")
println("Captured t3")

# Reset
state.calcDimsStruct.zoom = 1.0f0
state.calcDimsStruct.panX = 0.0f0
state.calcDimsStruct.panY = 0.0f0

# ============== TEST 4: Bone overlay ===============
println("\n--- TEST 4: Bone overlay ---")
# Move to a slice with bone (vertebral body)
put!(ch, 0); sleep(1)
capture_synced(ch, "/workspaces/MedEye3d.jl/feat_t4_bone.png")
println("Captured t4")

# ============== TEST 5: Bone window ===============
println("\n--- TEST 5: Bone window ---")
state.mainForDisplayObjects.listOfTextSpecifications[1].minAndMaxValue .= Float32[-700, 1300]
put!(ch, 0); sleep(1)
capture_synced(ch, "/workspaces/MedEye3d.jl/feat_t5_bonewin.png")
println("Captured t5")
state.mainForDisplayObjects.listOfTextSpecifications[1].minAndMaxValue .= Float32[-150, 250]

# ============== TEST 6: Upper thorax slice ===============
println("\n--- TEST 6: Different anatomy (thorax) ---")
put!(ch, 15); sleep(2)
println("Slice: $(state.currentDisplayedSlice)")
capture_synced(ch, "/workspaces/MedEye3d.jl/feat_t6_thorax.png")
println("Captured t6")

# ============== TEST 7: Lower pelvis slice ===============
println("\n--- TEST 7: Lower anatomy ---")
put!(ch, -40); sleep(2)
println("Slice: $(state.currentDisplayedSlice)")
capture_synced(ch, "/workspaces/MedEye3d.jl/feat_t7_pelvis.png")
println("Captured t7")

println("\n=== ALL FEATURE TESTS COMPLETE ===")
