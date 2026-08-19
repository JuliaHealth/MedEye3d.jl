using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using MedEye3d
using MedEye3d.BoneSubsegmentation
using KernelAbstractions

println("Testing KernelAbstractions BoneSubsegmentation...")

# Synthetic test volume (64x64x64)
dims = (64, 64, 64)
mask = zeros(Float32, dims)
bone = zeros(Float32, dims)

# Create lesion at center (32,32,32) radius 4
for z in 1:64, y in 1:64, x in 1:64
    d = sqrt((z-32)^2 + (y-32)^2 + (x-32)^2)
    if d <= 4.0
        mask[z, y, x] = 1.0f0
    end
    # Create bone shell around lesion (radius 6 to 12)
    if d >= 6.0 && d <= 12.0
        if d >= 10.0
            bone[z, y, x] = 2.0f0 # Cortical
        else
            bone[z, y, x] = 1.0f0 # Marrow
        end
    end
end

spacing = (1.0, 1.0, 1.0)
surf, marr = generate_bone_subsegments(mask, bone, spacing, 1)

println("KernelAbstractions execution successful!")
println("Surface voxels: $(count(surf))")
println("Marrow voxels: $(count(marr))")

@assert count(surf) > 0 "Bone surface should have voxels"
@assert count(marr) > 0 "Bone marrow should have voxels"
println("All assertions passed!")
