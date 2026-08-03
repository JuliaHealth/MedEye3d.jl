using Pkg

Pkg.instantiate()

using MedEye3d
using MedEye3d.ForDisplayStructs
using MedEye3d.MakieEvents
using MedImages
using ColorTypes
using GLMakie
using MedEye3d.SegmentationDisplay
using Setfield

# Define paths
data_dir = "/workspaces/MedEye3d.jl/data/pat6_pet_only_debug/pat6_pet_only_debug"

println("Loading data natively from MRB/MRML...")
loaded_nodes = MedImages.load_mrb(data_dir)

println("Extracting TP1 data...")
ct_1 = loaded_nodes["Fixed_CT_Volume_0"]
pet_1 = loaded_nodes["SUV_PET_Image_0"]
seg_1 = loaded_nodes["NNUNET_OR_Lesions_CT_0_artficials_added"]

println("Extracting TP2 data...")
ct_2 = loaded_nodes["Fixed_CT_Volume_1"]
pet_2 = loaded_nodes["SUV_PET_Image_1"]
seg_2 = loaded_nodes["NNUNET_OR_Lesions_CT_1_artficials_added"]



# MedImages resample_to_image now supports arbitrary rotation matrices.

# Use the exported resample function from MedImages (which MedEye3d relies on)
using CUDA
using Setfield

function gpu_resample(moving::MedImage, fixed::MedImage, interpolator)
    # 1. Allocate to GPU
    moving_cu = CuArray(moving.voxel_data)
    fixed_cu = CuArray(fixed.voxel_data)
    
    # 2. Create shallow GPU-based MedImages using Setfield to preserve all metadata
    moving_gpu_med = @set moving.voxel_data = moving_cu
    fixed_gpu_med = @set fixed.voxel_data = fixed_cu
    
    # 3. Perform fast GPU resampling
    # Note: resample_to_image expects (im_fixed, im_moving)
    resampled_gpu = MedImages.Resample_to_target.resample_to_image(fixed_gpu_med, moving_gpu_med, interpolator)
    
    # 4. Bring result back to CPU
    resampled_cpu_arr = Array(resampled_gpu.voxel_data)
    
    # 5. Explicitly free GPU memory to prevent GC / LLVM crashes
    CUDA.unsafe_free!(moving_cu)
    CUDA.unsafe_free!(fixed_cu)
    CUDA.unsafe_free!(resampled_gpu.voxel_data)
    
    # 6. Return reconstructed CPU MedImage
    return @set resampled_gpu.voxel_data = resampled_cpu_arr
end

if size(pet_1.voxel_data) != size(ct_1.voxel_data)
    pet_1 = gpu_resample(pet_1, ct_1, MedImages.MedImage_data_struct.B_spline_en)
end
if size(seg_1.voxel_data) != size(ct_1.voxel_data)
    seg_1 = gpu_resample(seg_1, ct_1, MedImages.MedImage_data_struct.Nearest_neighbour_en)
end

if size(pet_2.voxel_data) != size(ct_2.voxel_data)
    pet_2 = gpu_resample(pet_2, ct_2, MedImages.MedImage_data_struct.B_spline_en)
end
if size(seg_2.voxel_data) != size(ct_2.voxel_data)
    seg_2 = gpu_resample(seg_2, ct_2, MedImages.MedImage_data_struct.Nearest_neighbour_en)
end

# Apply MedEye3d's standard orientation flip
function correct_orientation!(med_img)
    med_img.voxel_data .= reverse(med_img.voxel_data, dims=2)
end

correct_orientation!(ct_1)
correct_orientation!(pet_1)
correct_orientation!(seg_1)

correct_orientation!(ct_2)
correct_orientation!(pet_2)
correct_orientation!(seg_2)


vol_ct_1 = Float32.(ct_1.voxel_data)
vol_pet_1 = Float32.(pet_1.voxel_data)
vol_seg_1 = Float32.(seg_1.voxel_data)

vol_ct_2 = Float32.(ct_2.voxel_data)
vol_pet_2 = Float32.(pet_2.voxel_data)
vol_seg_2 = Float32.(seg_2.voxel_data)

spacing = Tuple(Float64.(ct_1.spacing))

# Setup TextureSpecs
textureSpec_ct = TextureSpec{Float32}(
    name="CT",
    isMainImage=true,
    color=RGB(1.0, 1.0, 1.0),
    minAndMaxValue=Float32.([-150, 250]),
    maskContribution=1.0
)
textureSpec_pet = TextureSpec{Float32}(
    name="PET",
    isMainImage=false,
    isContinuusMask=true,
    colorSet=[RGB(0.0, 0.0, 0.0), RGB(1.0, 0.0, 0.0), RGB(1.0, 1.0, 0.0)],
    minAndMaxValue=Float32.([0, 15]),
    maskContribution=0.5
)
textureSpec_seg = TextureSpec{Float32}(
    name="Seg",
    isMainImage=false,
    isContinuusMask=false,
    color=RGB(0.0, 1.0, 0.0),
    minAndMaxValue=Float32.([1, 1000]),
    maskContribution=0.7
)

# Setup MultiImage
voxelDataTupleVector = Vector{Vector{Any}}([
    Any[("CT", vol_ct_1), ("PET", vol_pet_1), ("Seg", vol_seg_1)],
    Any[("CT", vol_ct_2), ("PET", vol_pet_2), ("Seg", vol_seg_2)]
])
spacing_1 = Tuple(Float64.(ct_1.spacing))
spacing_2 = Tuple(Float64.(ct_2.spacing))
origin_1 = Tuple(Float64.(ct_1.origin))
origin_2 = Tuple(Float64.(ct_2.origin))

# 4) Setup OpenGL Textures
textureSpecArray = Vector{Vector{TextureSpec}}([
    TextureSpec[deepcopy(textureSpec_ct), deepcopy(textureSpec_pet), deepcopy(textureSpec_seg)],
    TextureSpec[deepcopy(textureSpec_ct), deepcopy(textureSpec_pet), deepcopy(textureSpec_seg)]
])

spacings = [[spacing_1], [spacing_2]]
origins = [[origin_1], [origin_2]]
svVertAndInd = Dict{String, Vector}()
dummyStudySrc = Vector{Vector{Tuple{String,String}}}()

println("Starting MedEye3d Viewer...")
mainMedEye3dInstance = SegmentationDisplay.displayImage(
    dummyStudySrc;
    textureSpecArray=textureSpecArray,
    voxelDataTupleVector=voxelDataTupleVector,
    spacings=spacings,
    origins=origins,
    fractionOfMainImage=Float32(1.0),
    windowWidth=1400,
    svVertAndInd=svVertAndInd,
    quadView=false
)

# Start Makie Control Window
println("Starting Makie Control Window...")
fig = Figure(size=(400, 300))
layout = fig[1, 1] = GridLayout()

ax_btn = Button(fig, label="Axial", buttoncolor=:lightblue)
cor_btn = Button(fig, label="Coronal", buttoncolor=:lightblue)
sag_btn = Button(fig, label="Sagittal", buttoncolor=:lightblue)
layout[1, 1:3] = [ax_btn, cor_btn, sag_btn]

compare_toggle = Toggle(fig, active=true)
layout[2, 1] = Label(fig, "Compare TP1/TP2:")
layout[2, 2] = compare_toggle

layout[3, 1] = Label(fig, "Single Lesion ID (0=All):")
lesion_slider = Slider(fig, range=0:20, startvalue=0)
layout[3, 2:3] = lesion_slider

# Event callbacks dispatching to mainChannel
on(ax_btn.clicks) do _
    put!(mainMedEye3dInstance.channel, ChangePlaneEvent(:Axial))
end
on(cor_btn.clicks) do _
    put!(mainMedEye3dInstance.channel, ChangePlaneEvent(:Coronal))
end
on(sag_btn.clicks) do _
    put!(mainMedEye3dInstance.channel, ChangePlaneEvent(:Sagittal))
end

on(compare_toggle.active) do active
    put!(mainMedEye3dInstance.channel, CompareTimePointsEvent(active))
end

on(lesion_slider.value) do val
    put!(mainMedEye3dInstance.channel, ShowSingleLesionEvent(val))
end

# Make TP2 invisible if compare is false initially
if !compare_toggle.active[]
    put!(mainMedEye3dInstance.channel, CompareTimePointsEvent(false))
end

screen = display(fig)

# Keep script alive
wait(screen)
println("Done.")
