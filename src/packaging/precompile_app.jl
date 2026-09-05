# precompile_app.jl
# Workload tracing script for PackageCompiler.jl
# Executes typical visualization workloads, allocates OpenGL/display structs,
# performs volume resamplings and colormap conversions to compile method instances into the sysimage.

println("=== Executing MedEye3D Precompilation Traces ===")

using MedEye3d
using MedEye3d.ForDisplayStructs
using MedEye3d.DataStructs
using MedEye3d.StructsManag
using MedEye3d.SegmentationDisplay
using MedEye3d.MakieEvents
using MedEye3d.StrokeRasterization
using MedEye3d.ConnectedComponents
using MedEye3d.AIInference
using MedEye3d.LesionMetadataWindow
using MedEye3d.LesionAssociation

using MedImages
using ColorTypes
using Statistics
using LinearAlgebra
using Dates
using GLMakie
import Observables
import GLFW
import HDF5
import JSON

# 1. Trace TextureSpec and Display struct creation
println("[1/5] Tracing TextureSpec & struct initialization...")
colors_mapped = map(c -> RGB(c[1]/255, c[2]/255, c[3]/255), MedEye3d.distinctColorsSaved.listOfColors)

textureSpec_ct = TextureSpec{Float32}(
    name="CT",
    isMainImage=true,
    color=RGB(1.0, 1.0, 1.0),
    minAndMaxValue=Float32.([-150, 250])
)

textureSpec_pet = TextureSpec{Float32}(
    name="PET",
    isMainImage=false,
    isNuclearMask=true,
    color=RGB(1.0, 0.5, 0.0),
    minAndMaxValue=Float32.([0, 10]),
    maskContribution=0.5f0
)

textureSpec_mask = TextureSpec{Int16}(
    name="Mask",
    isMainImage=false,
    isMultiDiscreteMask=true,
    isIntegerTexture=true,
    colorSet=colors_mapped,
    minAndMaxValue=Int16.([0, length(colors_mapped)]),
    isEditable=true,
    maskContribution=0.5f0
)

textureSpec_bone = TextureSpec{Int8}(
    name="Bone_Overlay",
    isMainImage=false,
    isIntegerTexture=true,
    color=RGB(0.0, 1.0, 1.0),
    minAndMaxValue=Int8.([0, 3]),
    isVisible=true,
    maskContribution=0.5f0
)

# 2. Trace 3D synthetic volume generation & memory representations
println("[2/5] Tracing 3D volume manipulation & multi-planar permutations...")
dim_x, dim_y, dim_z = 64, 64, 32
vol_ct = rand(Float32, dim_x, dim_y, dim_z)
vol_mask = zeros(Int16, dim_x, dim_y, dim_z)
vol_mask[20:40, 20:40, 10:20] .= Int16(1)

# Axial, Coronal, Sagittal permutations
vol_ct_coronal = PermutedDimsArray(vol_ct, (1, 3, 2))
vol_mask_coronal = PermutedDimsArray(vol_mask, (1, 3, 2))

vol_ct_sagittal = PermutedDimsArray(vol_ct, (2, 3, 1))
vol_mask_sagittal = PermutedDimsArray(vol_mask, (2, 3, 1))

spacing_axial = (1.0, 1.0, 2.0)
spacing_coronal = (1.0, 2.0, 1.0)
spacing_sagittal = (1.0, 2.0, 1.0)

# 3. Trace QuadView data packaging
println("[3/5] Tracing QuadView multi-panel payload construction...")
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

spacings = [[spacing_axial], [spacing_axial], [spacing_coronal], [spacing_sagittal]]
origins = [[(0.0, 0.0, 0.0)], [(0.0, 0.0, 0.0)], [(0.0, 0.0, 0.0)], [(0.0, 0.0, 0.0)]]
svVertAndInd = Dict{String, Vector}()
dummyStudySrc = Vector{Vector{Tuple{String,String}}}()

# 4. Trace MedImages and Subsegmentation pipelines
println("[4/5] Tracing MedImages & Subsegmentation algorithms...")
try
    dummy_origin = (0.0, 0.0, 0.0)
    dummy_dir = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    med_im = MedImage(
        voxel_data=vol_ct,
        spacing=spacing_axial,
        origin=dummy_origin,
        direction=dummy_dir,
        image_type=MedImages.MedImage_data_struct.CT_type,
        image_subtype=MedImages.MedImage_data_struct.CT_subtype,
        patient_id="precompile_trace"
    )
    # Trace bone subsegmentation computation
    synth_mask = zeros(Int16, dim_x, dim_y, dim_z)
    synth_mask[10:15, 10:15, 5:10] .= Int16(1)
    synth_skelly = zeros(Float32, dim_x, dim_y, dim_z)
    synth_skelly[8:18, 8:18, 4:12] .= Float32(2.0)
    synth_skelly[11:14, 11:14, 6:9] .= Float32(1.0)
    MedEye3d.SegmentationDisplay.MakieEventHandlers.compute_bone_subsegments_fast(synth_mask, synth_skelly, 1)
catch e
    println("MedImage tracing notice: ", e)
end

# 5. Trace AppMain entry dispatch & StudySelectorWindow
println("[5/5] Tracing AppMain & StudySelectorWindow...")
MedEye3d.StudySelectorWindow.scan_medical_files(@__DIR__)

# Trace Makie metadata control panel window layout
try
    active_les = Observables.Observable("1: Lesion_1")
    les_list = Observables.Observable(["1: Lesion_1", "2: Lesion_2"])
    makie_win = LesionMetadataWindow.create_metadata_window(active_les, les_list, nothing)
catch e
    println("Makie layout tracing notice: ", e)
end

include(joinpath(@__DIR__, "AppMain.jl"))
using .MedEye3dApp

# Invoke help & version dispatch paths
MedEye3dApp.run_app(["--help"])
MedEye3dApp.run_app(["--version"])

# Trace HDF5 opening in MedEye3dApp
try
    h5_path = joinpath(@__DIR__, "..", "..", "data", "preprocessed_volumes.h5")
    if isfile(h5_path)
        h5 = MedEye3dApp.HDF5.h5open(h5_path, "r")
        close(h5)
    end
catch e
end

println("=== MedEye3D Precompilation Traces Completed Successfully ===")
