using MedEye3d
using MedEye3d.ForDisplayStructs
using MedEye3d.DataStructs
using MedEye3d.StructsManag
using MedEye3d.SegmentationDisplay
using ColorTypes
using Statistics
using FileIO
using ImageIO
import GLFW
import ModernGL

println("=== MedEye3D Screenshot Verification Suite ===")

# Create synthetic 3D CT + Segmentation Mask
dim_x, dim_y, dim_z = 128, 128, 64
vol_ct = zeros(Float32, dim_x, dim_y, dim_z)
vol_mask = zeros(Float32, dim_x, dim_y, dim_z)

cx, cy, cz = dim_x ÷ 2, dim_y ÷ 2, dim_z ÷ 2
for z in 1:dim_z, y in 1:dim_y, x in 1:dim_x
    vol_ct[x, y, z] = -100.0f0 + 40.0f0 * sin(0.08f0 * x) * cos(0.08f0 * y)
    dx = (x - cx) / (dim_x * 0.35f0)
    dy = (y - cy) / (dim_y * 0.35f0)
    dz = (z - cz) / (dim_z * 0.35f0)
    r2 = dx*dx + dy*dy + dz*dz
    if r2 < 1.0f0
        vol_ct[x, y, z] = 40.0f0 + 120.0f0 * (1.0f0 - Float32(sqrt(r2)))
    end

    lx = (x - (cx + 16)) / 10.0f0
    ly = (y - (cy + 8)) / 10.0f0
    lz = (z - (cz - 4)) / 8.0f0
    if (lx*lx + ly*ly + lz*lz) < 1.0f0
        vol_mask[x, y, z] = 1.0f0
    end
end

spacing = (1.0, 1.0, 1.5)
vol_ct_coronal = permutedims(vol_ct, (1, 3, 2))
vol_mask_coronal = permutedims(vol_mask, (1, 3, 2))
spacing_coronal = (spacing[1], spacing[3], spacing[2])

vol_ct_sagittal = permutedims(vol_ct, (2, 3, 1))
vol_mask_sagittal = permutedims(vol_mask, (2, 3, 1))
spacing_sagittal = (spacing[2], spacing[3], spacing[1])

textureSpec_ct = TextureSpec{Float32}(
    name="CT",
    isMainImage=true,
    color=RGB(1.0, 1.0, 1.0),
    minAndMaxValue=Float32.([-150, 250])
)

textureSpec_mask = TextureSpec{Float32}(
    name="Mask",
    isMainImage=false,
    color=RGB(1.0, 0.2, 0.2),
    minAndMaxValue=Float32.([0, 1]),
    maskContribution=0.5f0,
    isEditable=true
)

voxelDataTupleVector = Vector{Vector{Any}}([
    Any[("CT", vol_ct), ("Mask", vol_mask)],
    Any[("Mask", vol_mask)],
    Any[("CT", vol_ct_coronal), ("Mask", vol_mask_coronal)],
    Any[("CT", vol_ct_sagittal), ("Mask", vol_mask_sagittal)]
])

textureSpecArray = Vector{Vector{TextureSpec}}([
    TextureSpec[deepcopy(textureSpec_ct), deepcopy(textureSpec_mask)],
    TextureSpec[deepcopy(textureSpec_mask)],
    TextureSpec[deepcopy(textureSpec_ct), deepcopy(textureSpec_mask)],
    TextureSpec[deepcopy(textureSpec_ct), deepcopy(textureSpec_mask)]
])

spacings = [[spacing], [spacing], [spacing_coronal], [spacing_sagittal]]
origins = [[(0.0, 0.0, 0.0)], [(0.0, 0.0, 0.0)], [(0.0, 0.0, 0.0)], [(0.0, 0.0, 0.0)]]
svVertAndInd = Dict{String, Vector}()
dummyStudySrc = Vector{Vector{Tuple{String,String}}}()

win_w, win_h = 1280, 1280
println("Initializing QuadView window ($win_w x $win_h)...")
mainViewer = SegmentationDisplay.displayImage(
    dummyStudySrc;
    textureSpecArray=textureSpecArray,
    voxelDataTupleVector=voxelDataTupleVector,
    spacings=spacings,
    origins=origins,
    fractionOfMainImage=Float32(1.0),
    windowWidth=win_w,
    svVertAndInd=svVertAndInd,
    quadView=true
)

println("Rendering frames...")
# Allow OpenGL rendering loop to process initial render passes
sleep(2.0)

# Scroll to center slice to show lesion
for diff in [16, 0]
    put!(mainViewer.channel, Int64(diff))
    sleep(0.5)
end

println("Grabbing OpenGL framebuffer via glReadPixels...")
# Get the GLFW window
gl_win = mainViewer.states[1].mainForDisplayObjects.window
GLFW.MakeContextCurrent(gl_win)

raw_pixels = Array{UInt8}(undef, 3, win_w, win_h)
ModernGL.glReadPixels(0, 0, win_w, win_h, ModernGL.GL_RGB, ModernGL.GL_UNSIGNED_BYTE, raw_pixels)

# Convert to Image matrix (Height x Width) with Y-flipped
img_out = Matrix{RGB{Float32}}(undef, win_h, win_w)
for y in 1:win_h
    # OpenGL (0,0) is bottom-left; Image (1,1) is top-left
    src_y = win_h - y + 1
    for x in 1:win_w
        r = Float32(raw_pixels[1, x, src_y]) / 255.0f0
        g = Float32(raw_pixels[2, x, src_y]) / 255.0f0
        b = Float32(raw_pixels[3, x, src_y]) / 255.0f0
        img_out[y, x] = RGB{Float32}(r, g, b)
    end
end

output_dir = normpath(joinpath(@__DIR__, "..", "screenshots"))
mkpath(output_dir)
output_file = joinpath(output_dir, "medeye3d_quadview_render.png")
FileIO.save(output_file, img_out)
println("✓ Screenshot saved successfully: $output_file")

# Close window cleanly
try
    put!(mainViewer.channel, MedEye3d.ForDisplayStructs.CloseWindowEvent())
    sleep(0.5)
catch
end
println("=== Screenshot Verification Completed Successfully ===")
