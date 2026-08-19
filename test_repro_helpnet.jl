using HDF5
using MedEye3d
using MedEye3d.MakieEvents
using Statistics

println("\n[Repro] === REPRODUCE HELPNET + NNINTERACTIVE ISSUES ===")

data_dir = "/mnt/big/project_ssd/project_ssd/MedEye3d.jl/data/pat_6_files"
h5 = HDF5.h5open(joinpath(data_dir, "preprocessed_volumes.h5"), "r")
ct_vol = Float32.(read(h5["BASELINE/Fixed_CT_Volume_0.nii.gz"]))
pet_vol = Float32.(read(h5["BASELINE/SUV_PET_Image_0.nii.gz"]))
mask_vol = Float32.(read(h5["BASELINE/PET_Lesions_0.nii.gz"]))
close(h5)

ct_base = reverse(ct_vol, dims=2)
pet_base = reverse(pet_vol, dims=2)
mask_base = reverse(mask_vol, dims=2)

# Find a real lesion center from the mask
lesion_ids = unique(mask_base[mask_base .> 0])
println("[Repro] Found $(length(lesion_ids)) lesion IDs: $lesion_ids")

# Use lesion_id = 1 (first lesion)
lid = Float32(lesion_ids[1])
pts = findall(mask_base .== lid)
cx = round(Int, mean([p[1] for p in pts]))
cy = round(Int, mean([p[2] for p in pts]))
cz = round(Int, mean([p[3] for p in pts]))
println("[Repro] Lesion $lid center: ($cx, $cy, $cz), $(length(pts)) voxels")

# Simulate user painting: create scribbles at lesion center (like a real user would paint)
points_vol = zeros(Float32, size(ct_base))
for dx in -3:3, dy in -3:3
    x, y, z = cx+dx, cy+dy, cz
    if checkbounds(Bool, points_vol, x, y, z)
        points_vol[x, y, z] = 1.0f0
    end
end
scribble_count = count(points_vol .> 0)
println("[Repro] Created $scribble_count scribble voxels at ($cx, $cy, $cz)")

# === TEST 1: HELPNet with SCRIBBLE (current behavior) ===
println("\n[Repro] TEST 1: HELPNet with FULL SCRIBBLE (current behavior)...")
t1 = time()
mask1 = MedEye3d.InferenceClient.run_helpnet_inference(ct_base, pet_base, points_vol, cx, cy, cz)
t1_elapsed = time() - t1
if mask1 !== nothing
    v1 = count(mask1 .> 0)
    println("[Repro] HELPNet (scribble): $v1 voxels in $(round(t1_elapsed, digits=2))s")
else
    println("[Repro] HELPNet (scribble): FAILED (returned nothing)")
end

# === TEST 2: HELPNet with SINGLE CENTER POINT (what user says works) ===
println("\n[Repro] TEST 2: HELPNet with SINGLE CENTER POINT...")
center_points = zeros(Float32, size(ct_base))
center_points[cx, cy, cz] = 1.0f0
t2 = time()
mask2 = MedEye3d.InferenceClient.run_helpnet_inference(ct_base, pet_base, center_points, cx, cy, cz)
t2_elapsed = time() - t2
if mask2 !== nothing
    v2 = count(mask2 .> 0)
    println("[Repro] HELPNet (center point): $v2 voxels in $(round(t2_elapsed, digits=2))s")
else
    println("[Repro] HELPNet (center point): FAILED (returned nothing)")
end

# === TEST 3: NNInteractive with SCRIBBLE ===
println("\n[Repro] TEST 3: NNInteractive with scribble...")
t3 = time()
mask3 = MedEye3d.InferenceClient.run_nninteractive(ct_base, pet_base, points_vol, cx, cy, cz)
t3_elapsed = time() - t3
if mask3 !== nothing
    v3 = count(mask3 .> 0)
    println("[Repro] NNInteractive: $v3 voxels in $(round(t3_elapsed, digits=2))s")
else
    println("[Repro] NNInteractive: FAILED (returned nothing)")
end

println("\n[Repro] === RESULTS COMPARISON ===")
println("[Repro] HELPNet (scribble):     $(mask1 !== nothing ? count(mask1 .> 0) : "FAILED") voxels, $(round(t1_elapsed, digits=2))s")
println("[Repro] HELPNet (center point): $(mask2 !== nothing ? count(mask2 .> 0) : "FAILED") voxels, $(round(t2_elapsed, digits=2))s")
println("[Repro] NNInteractive:          $(mask3 !== nothing ? count(mask3 .> 0) : "FAILED") voxels, $(round(t3_elapsed, digits=2))s")
