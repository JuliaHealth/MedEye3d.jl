using NIfTI

# Dimensions and spacing
nx, ny, nz = 128, 128, 128
spacing = (1.0, 1.0, 1.0)

# Create volume
vol = zeros(Float32, nx, ny, nz)
cx, cy, cz = 64.0, 64.0, 64.0
radius_physical = 40.0

for i in 1:nx, j in 1:ny, k in 1:nz
    dx = (i - cx) * spacing[1]
    dy = (j - cy) * spacing[2]
    dz = (k - cz) * spacing[3]
    if sqrt(dx^2 + dy^2 + dz^2) <= radius_physical
        vol[i, j, k] = 1.0f0
    end
end

# Save to NIfTI
ni = NIVolume(vol)
# Set spacing in NIfTI header
ni.header.pixdim = (1.0, spacing[1], spacing[2], spacing[3], 0.0, 0.0, 0.0, 0.0)
ni.header.qform_code = 1
ni.header.sform_code = 1

niwrite("data/synthetic_sphere.nii", ni)
println("Saved data/synthetic_sphere.nii with spacing \$spacing")
