# FULL END-TO-END TEST through on_next! multiple dispatch
# Tests: HELPNet (center point), NNInteractive, inference queue, status Observable
using HDF5, ModernGL, GLFW, GLMakie
using MedEye3d, MedEye3d.ForDisplayStructs, MedEye3d.DataStructs
using MedEye3d.MakieEvents, MedEye3d.SegmentationDisplay
using Statistics, Observables

MEH = MedEye3d.SegmentationDisplay.MakieEventHandlers

println("\n=== FULL ON_NEXT! DISPATCH E2E TEST ===")
println("Tests: HELPNet center point, NNInteractive, queue, status, GPU")

# Verify GPU
println("\n[1/7] GPU Check...")
run(`docker exec medeye3d-ai python3 -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"`)

# Verify status Observable
println("\n[2/7] Status Observable: $(MEH.ai_status_text[])")
@assert MEH.ai_status_text[] == "Ready"

# Load real patient data
data_dir = "/mnt/big/project_ssd/project_ssd/MedEye3d.jl/data/pat_6_files"
h5 = HDF5.h5open(joinpath(data_dir, "preprocessed_volumes.h5"), "r")
ct_vol = Float32.(read(h5["BASELINE/Fixed_CT_Volume_0.nii.gz"]))
pet_vol = Float32.(read(h5["BASELINE/SUV_PET_Image_0.nii.gz"]))
mask_vol = Float32.(read(h5["BASELINE/PET_Lesions_0.nii.gz"]))
sp = Tuple(Float64.(read(HDF5.attributes(h5["BASELINE/Fixed_CT_Volume_0.nii.gz"])["spacing"])))
close(h5)

ct = reverse(ct_vol, dims=2); pet = reverse(pet_vol, dims=2); mask = reverse(mask_vol, dims=2)

# Build quad-view panel data
surf = zeros(Float32, size(ct)); marr = zeros(Float32, size(ct))
vdt = [
    Any[("CT",ct),("PET",pet),("Mask",copy(mask)),("Bone_Surface",surf),("Bone_Marrow",marr)],
    Any[("CT",ct),("PET",pet),("Mask",copy(mask)),("Bone_Surface",copy(surf)),("Bone_Marrow",copy(marr))],
    Any[("CT",permutedims(ct,(2,3,1))),("PET",permutedims(pet,(2,3,1))),("Mask",copy(permutedims(mask,(2,3,1)))),("Bone_Surface",zeros(Float32,size(permutedims(ct,(2,3,1))))),("Bone_Marrow",zeros(Float32,size(permutedims(ct,(2,3,1)))))],
    Any[("CT",permutedims(ct,(1,3,2))),("PET",permutedims(pet,(1,3,2))),("Mask",copy(permutedims(mask,(1,3,2)))),("Bone_Surface",zeros(Float32,size(permutedims(ct,(1,3,2))))),("Bone_Marrow",zeros(Float32,size(permutedims(ct,(1,3,2)))))],
]
tsa = Vector{Vector{TextureSpec}}([
    TextureSpec[
        TextureSpec{Float32}(name="CT",isMainImage=true,minAndMaxValue=Float32.([0,100])),
        TextureSpec{Float32}(name="PET",isMainImage=true,minAndMaxValue=Float32.([0,100])),
        TextureSpec{Float32}(name="Mask",isMultiDiscreteMask=true,isEditable=true,minAndMaxValue=Float32.([0,1000])),
        TextureSpec{Float32}(name="Bone_Surface",isContinuusMask=true,minAndMaxValue=Float32.([0,1])),
        TextureSpec{Float32}(name="Bone_Marrow",isContinuusMask=true,minAndMaxValue=Float32.([0,1])),
    ],
    TextureSpec[
        TextureSpec{Float32}(name="CT",isMainImage=true,minAndMaxValue=Float32.([0,100])),
        TextureSpec{Float32}(name="PET",isMainImage=true,minAndMaxValue=Float32.([0,100])),
        TextureSpec{Float32}(name="Mask",isMultiDiscreteMask=true,isEditable=true,minAndMaxValue=Float32.([0,1000])),
        TextureSpec{Float32}(name="Bone_Surface",isContinuusMask=true,minAndMaxValue=Float32.([0,1])),
        TextureSpec{Float32}(name="Bone_Marrow",isContinuusMask=true,minAndMaxValue=Float32.([0,1])),
    ],
    TextureSpec[
        TextureSpec{Float32}(name="CT",isMainImage=true,minAndMaxValue=Float32.([0,100])),
        TextureSpec{Float32}(name="PET",isMainImage=true,minAndMaxValue=Float32.([0,100])),
        TextureSpec{Float32}(name="Mask",isMultiDiscreteMask=true,isEditable=true,minAndMaxValue=Float32.([0,1000])),
        TextureSpec{Float32}(name="Bone_Surface",isContinuusMask=true,minAndMaxValue=Float32.([0,1])),
        TextureSpec{Float32}(name="Bone_Marrow",isContinuusMask=true,minAndMaxValue=Float32.([0,1])),
    ],
    TextureSpec[
        TextureSpec{Float32}(name="CT",isMainImage=true,minAndMaxValue=Float32.([0,100])),
        TextureSpec{Float32}(name="PET",isMainImage=true,minAndMaxValue=Float32.([0,100])),
        TextureSpec{Float32}(name="Mask",isMultiDiscreteMask=true,isEditable=true,minAndMaxValue=Float32.([0,1000])),
        TextureSpec{Float32}(name="Bone_Surface",isContinuusMask=true,minAndMaxValue=Float32.([0,1])),
        TextureSpec{Float32}(name="Bone_Marrow",isContinuusMask=true,minAndMaxValue=Float32.([0,1])),
    ],
])
spacings = [[sp],[sp],[(sp[2],sp[3],sp[1])],[(sp[1],sp[3],sp[2])]]
origins = [[(0.0,0.0,0.0)] for _ in 1:4]

println("\n[3/7] Launching QuadView viewer...")
inst = SegmentationDisplay.displayImage(Vector{Vector{Tuple{String,String}}}();
    textureSpecArray=tsa, voxelDataTupleVector=vdt, spacings=spacings, origins=origins, quadView=true)
ch = inst.channel
sleep(3)

# Activate painting
put!(ch, PaintValEvent(32, true)); sleep(0.3)

# Paint scribbles at a real lesion center
pts = findall(mask .== 1.0f0)
cx = round(Int, mean([p[1] for p in pts]))
cy = round(Int, mean([p[2] for p in pts]))
cz = round(Int, mean([p[3] for p in pts]))
println("[4/7] Painting at lesion center ($cx,$cy,$cz)...")

# Paint into manualModif (auto-inserted at index 2)
mm = vdt[1][2][2]  # manualModif volume
for dx in -3:3, dy in -3:3
    x, y = cx+dx, cy+dy
    checkbounds(Bool, mm, x, y, cz) && (mm[x, y, cz] = 32.0f0)
end
println("   Painted $(count(mm .> 0)) voxels into manualModif")

# Get seg_vol reference
seg = nothing
for e in vdt[1]; e[1] == "Mask" && (seg = e[2]; break); end
before = count(seg .== 32.0f0)
println("   seg_vol label=32 BEFORE: $before")

# ===== TEST: HELPNet via on_next! dispatch =====
println("\n[5/7] TEST: HELPNet via on_next! AddAutoPetEvent → queue → Docker → AIInferenceResultEvent...")
put!(ch, AddAutoPetEvent("HELPNet (AI)", ch))
for i in 1:30
    sleep(1)
    c = count(seg .== 32.0f0)
    if c > before
        println("   HELPNet DETECTED at $(i)s: $(c - before) new voxels, status: $(MEH.ai_status_text[])")
        break
    end
    i % 5 == 0 && println("   ... $(i)s, status: $(MEH.ai_status_text[])")
end
helpnet_count = count(seg .== 32.0f0)
println("   HELPNet result: $(helpnet_count - before) new voxels")

# Reset manualModif for nnInteractive test
mm .= 0.0f0
for dx in -3:3, dy in -3:3
    x, y = cx+dx, cy+dy
    checkbounds(Bool, mm, x, y, cz) && (mm[x, y, cz] = 33.0f0)
end

# ===== TEST: NNInteractive via on_next! dispatch =====
before_nn = count(seg .== 33.0f0)
println("\n[6/7] TEST: NNInteractive via on_next! AddAutoPetEvent → queue → Docker → AIInferenceResultEvent...")

# Need to update active lesion ID
put!(ch, PaintValEvent(33, true)); sleep(0.3)
put!(ch, AddAutoPetEvent("NNInteractive", ch))
for i in 1:30
    sleep(1)
    c = count(seg .== 33.0f0)
    if c > before_nn
        println("   NNInteractive DETECTED at $(i)s: $(c - before_nn) new voxels, status: $(MEH.ai_status_text[])")
        break
    end
    i % 5 == 0 && println("   ... $(i)s, status: $(MEH.ai_status_text[])")
end
nn_count = count(seg .== 33.0f0)
println("   NNInteractive result: $(nn_count - before_nn) new voxels")

# Final summary
println("\n[7/7] === FINAL RESULTS ===")
println("   HELPNet (center point, via on_next!):  $(helpnet_count - before) voxels")
println("   NNInteractive (scribble, via on_next!): $(nn_count - before_nn) voxels")
println("   Status Observable final: $(MEH.ai_status_text[])")

if (helpnet_count - before) > 0 && (nn_count - before_nn) > 0
    println("\n   ✅ ALL TESTS PASSED — both models work through on_next! dispatch")
else
    println("\n   ❌ TESTS FAILED")
    (helpnet_count - before) == 0 && println("      HELPNet returned 0 voxels!")
    (nn_count - before_nn) == 0 && println("      NNInteractive returned 0 voxels!")
end
