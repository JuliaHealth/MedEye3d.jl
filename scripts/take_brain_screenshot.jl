using MedEye3d
using MedImages
using ColorTypes
using Statistics
using FileIO
using ImageIO
using MedEye3d.ForDisplayStructs
using MedEye3d.DataStructs
using MedEye3d.StructsManag
using MedEye3d.SegmentationDisplay
import GLFW
import ModernGL

brain_path = "D:\\MedEye3d.jl\\build\\MedEye3D_dist\\share\\julia\\artifacts\\ad4e594b35357bcfafa2ed97db3137382a3f09bb\\brain.nii.gz"
println("Loading brain image: $brain_path")

med_img = MedImages.load_image(brain_path, "")
vol_img = Float32.(med_img.voxel_data)
spacing = Tuple(Float64.(med_img.spacing))
origin = Tuple(Float64.(med_img.origin))

# Synthetic segmentation mask on central brain region
vol_mask = zeros(Float32, size(vol_img)...)
cx, cy, cz = size(vol_img) .÷ 2
for z in max(1, cz-15):min(size(vol_img, 3), cz+15)
    for y in max(1, cy-20):min(size(vol_img, 2), cy+20)
        for x in max(1, cx-20):min(size(vol_img, 1), cx+20)
            dx = (x - cx) / 18.0f0
            dy = (y - cy) / 18.0f0
            dz = (z - cz) / 12.0f0
            if dx*dx + dy*dy + dz*dz < 1.0f0 && vol_img[x, y, z] > 50.0f0
                vol_mask[x, y, z] = 1.0f0
            end
        end
    end
end

min_val, max_val = Float32(minimum(vol_img)), Float32(maximum(vol_img))
println("Brain volume size: $(size(vol_img)), range: [$min_val, $max_val]")

textureSpec_img = TextureSpec{Float32}(
    name="MainImage",
    isMainImage=true,
    color=RGB(1.0, 1.0, 1.0),
    minAndMaxValue=Float32.([min_val, max_val])
)

textureSpec_mask = TextureSpec{Float32}(
    name="Mask",
    isMainImage=false,
    color=RGB(1.0, 0.1, 0.1),
    minAndMaxValue=Float32.([0, 1]),
    maskContribution=0.6f0,
    isEditable=true
)

# Multi-planar permutations
vol_img_coronal = permutedims(vol_img, (1, 3, 2))
vol_mask_coronal = permutedims(vol_mask, (1, 3, 2))
spacing_coronal = (spacing[1], spacing[3], spacing[2])

vol_img_sagittal = permutedims(vol_img, (2, 3, 1))
vol_mask_sagittal = permutedims(vol_mask, (2, 3, 1))
spacing_sagittal = (spacing[2], spacing[3], spacing[1])

voxelDataTupleVector = Vector{Vector{Any}}([
    Any[("MainImage", vol_img), ("Mask", vol_mask)],
    Any[("Mask", vol_mask)],
    Any[("MainImage", vol_img_coronal), ("Mask", vol_mask_coronal)],
    Any[("MainImage", vol_img_sagittal), ("Mask", vol_mask_sagittal)]
])

textureSpecArray = Vector{Vector{TextureSpec}}([
    TextureSpec[deepcopy(textureSpec_img), deepcopy(textureSpec_mask)],
    TextureSpec[deepcopy(textureSpec_mask)],
    TextureSpec[deepcopy(textureSpec_img), deepcopy(textureSpec_mask)],
    TextureSpec[deepcopy(textureSpec_img), deepcopy(textureSpec_mask)]
])

spacings = [[spacing], [spacing], [spacing_coronal], [spacing_sagittal]]
origins = [[origin], [origin], [origin], [origin]]
svVertAndInd = Dict{String, Vector}()
dummyStudySrc = Vector{Vector{Tuple{String,String}}}()

win_w, win_h = 1280, 1280
println("Opening MedEye3D Brain Visualizer ($win_w x $win_h)...")
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

sleep(2.0)
# Scroll to mid-slice (size ÷ 2)
mid_slice = size(vol_img, 3) ÷ 2
println("Scrolling to middle brain slice: $mid_slice")
put!(mainViewer.channel, Int64(mid_slice))
sleep(0.8)
# Trigger refresh
put!(mainViewer.channel, Int64(0))
sleep(0.5)

gl_win = mainViewer.states[1].mainForDisplayObjects.window
GLFW.MakeContextCurrent(gl_win)

raw_pixels = Array{UInt8}(undef, 3, win_w, win_h)
ModernGL.glReadPixels(0, 0, win_w, win_h, ModernGL.GL_RGB, ModernGL.GL_UNSIGNED_BYTE, raw_pixels)

img_out = Matrix{RGB{Float32}}(undef, win_h, win_w)
for y in 1:win_h
    src_y = win_h - y + 1
    for x in 1:win_w
        r = Float32(raw_pixels[1, x, src_y]) / 255.0f0
        g = Float32(raw_pixels[2, x, src_y]) / 255.0f0
        b = Float32(raw_pixels[3, x, src_y]) / 255.0f0
        img_out[y, x] = RGB{Float32}(r, g, b)
    end
end

output_file = "D:\\MedEye3d.jl\\screenshots\\medeye3d_brain_quadview.png"
FileIO.save(output_file, img_out)
println("✓ Brain screenshot saved: $output_file")

try
    put!(mainViewer.channel, MedEye3d.ForDisplayStructs.CloseWindowEvent())
    sleep(0.5)
catch
end
