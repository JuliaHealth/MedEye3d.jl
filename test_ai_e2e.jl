using HDF5
using ModernGL
using GLFW
using GLMakie
using MedEye3d
using MedEye3d.ForDisplayStructs
using MedEye3d.DataStructs
using MedEye3d.MakieEvents
using MedEye3d.SegmentationDisplay
using Statistics

println("\n[Test] === ASYNC AI INFERENCE E2E TEST ===")

data_dir_pat6 = "/mnt/big/project_ssd/project_ssd/MedEye3d.jl/data/pat_6_files"
cache_h5_file = joinpath(data_dir_pat6, "preprocessed_volumes.h5")

println("[Test] Reading patient data...")
h5_file = HDF5.h5open(cache_h5_file, "r")
ct_vol = Float32.(read(h5_file["BASELINE/Fixed_CT_Volume_0.nii.gz"]))
pet_vol = Float32.(read(h5_file["BASELINE/SUV_PET_Image_0.nii.gz"]))
first_spacing = Tuple(Float64.(read(HDF5.attributes(h5_file["BASELINE/Fixed_CT_Volume_0.nii.gz"])["spacing"])))
close(h5_file)

ct_vol_base = reverse(ct_vol, dims=2)
pet_vol_base = reverse(pet_vol, dims=2)

# === TEST 1: insert_patch! @views fix ===
println("\n[Test] TEST 1: insert_patch! @views fix...")
seg_vol_test = zeros(Float32, size(ct_vol_base))
test_patch = ones(UInt8, 64, 64, 64)
cx, cy, cz = 256, 256, 163
MedEye3d.InferenceClient.insert_patch!(seg_vol_test, test_patch, cx, cy, cz, label_val=32.0f0)
inserted_count = count(seg_vol_test .== 32.0f0)
println("[Test] insert_patch! wrote $inserted_count voxels (expected: $(64*64*64))")
if inserted_count != 64*64*64
    error("FAIL: insert_patch! did not write the expected number of voxels!")
end
println("[Test] SUCCESS: insert_patch! @views fix works")

# === TEST 2: HELPNet Docker round-trip ===
println("\n[Test] TEST 2: HELPNet Docker round-trip...")
points_vol_test = zeros(Float32, size(ct_vol_base))
for dx in -2:2, dy in -2:2, dz in -2:2
    points_vol_test[cx+dx, cy+dy, cz+dz] = 1.0f0
end
scribble_count = count(points_vol_test .> 0)
println("[Test] Created $scribble_count scribble voxels at ($cx, $cy, $cz)")

seg_vol_helpnet = zeros(Float32, size(ct_vol_base))
mask_helpnet = MedEye3d.InferenceClient.run_helpnet_inference(ct_vol_base, pet_vol_base, points_vol_test, cx, cy, cz)
if mask_helpnet === nothing
    error("FAIL: HELPNet Docker inference returned nothing!")
end
mask_voxels = count(mask_helpnet .> 0)
println("[Test] HELPNet Docker returned mask with $mask_voxels voxels")

MedEye3d.InferenceClient.insert_patch!(seg_vol_helpnet, mask_helpnet, cx, cy, cz, label_val=32.0f0)
seg_voxels = count(seg_vol_helpnet .== 32.0f0)
println("[Test] After insert_patch!, seg_vol has $seg_voxels voxels with label 32")
if seg_voxels == 0
    error("FAIL: HELPNet segmentation was not written to volume!")
end
println("[Test] SUCCESS: HELPNet works")

# === TEST 3: NNInteractive Docker round-trip ===
println("\n[Test] TEST 3: NNInteractive Docker round-trip...")
seg_vol_nn = zeros(Float32, size(ct_vol_base))
mask_nn = MedEye3d.InferenceClient.run_nninteractive(ct_vol_base, pet_vol_base, points_vol_test, cx, cy, cz)
if mask_nn === nothing
    error("FAIL: NNInteractive Docker inference returned nothing!")
end
nn_mask_voxels = count(mask_nn .> 0)
println("[Test] NNInteractive Docker returned mask with $nn_mask_voxels voxels")

MedEye3d.InferenceClient.insert_patch!(seg_vol_nn, mask_nn, cx, cy, cz, label_val=33.0f0)
nn_seg_voxels = count(seg_vol_nn .== 33.0f0)
println("[Test] After insert_patch!, seg_vol has $nn_seg_voxels voxels with label 33")
if nn_seg_voxels == 0
    error("FAIL: NNInteractive segmentation was not written to volume!")
end
println("[Test] SUCCESS: NNInteractive works")

# === TEST 4: AddAutoPetEvent struct has channel field ===
println("\n[Test] TEST 4: AddAutoPetEvent struct has channel field...")
test_ch = Channel{Any}(32)
evt = AddAutoPetEvent("HELPNet (AI)", test_ch)
println("[Test] Created AddAutoPetEvent with algorithm=$(evt.algorithm), channel type=$(typeof(evt.channel))")
println("[Test] SUCCESS: AddAutoPetEvent struct is correct")

# === TEST 5: AIInferenceResultEvent struct ===
println("\n[Test] TEST 5: AIInferenceResultEvent struct...")
dummy_mask = ones(UInt8, 10, 10, 10)
dummy_seg = zeros(Float32, 100, 100, 100)
result_evt = AIInferenceResultEvent("HELPNet (AI)", 32, 50, 50, 50, dummy_mask, dummy_seg)
println("[Test] Created AIInferenceResultEvent: algo=$(result_evt.algorithm), id=$(result_evt.active_id), seed=($(result_evt.cx),$(result_evt.cy),$(result_evt.cz))")
println("[Test] SUCCESS: AIInferenceResultEvent struct is correct")

println("\n[Test] === ALL TESTS PASSED ===")
println("[Test] Summary:")
println("[Test]   ✓ insert_patch! @views fix verified")
println("[Test]   ✓ HELPNet Docker: $mask_voxels predicted, $seg_voxels written")
println("[Test]   ✓ NNInteractive Docker: $nn_mask_voxels predicted, $nn_seg_voxels written")
println("[Test]   ✓ AddAutoPetEvent has channel field")
println("[Test]   ✓ AIInferenceResultEvent struct works")
