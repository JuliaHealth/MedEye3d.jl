using MedEye3d
using HDF5, NIfTI
#Data downloaded from the following google drive links :

# https://drive.google.com/file/d/1Segr6BC_ma9bKNmM8lzUBaJLQChkAWKA/view?usp=drive_link
# https://drive.google.com/file/d/1PqHTQXOVTWx0FQSDE0oH4hRKNCE9jCx8/view?usp=drive_link
#modify paths to your downloaded data accordingly

ctNiftiImage = "D:/mingw_installation/home/hurtbadly/Downloads/ct_soft_pat_3_sudy_0.nii.gz"
petNiftiImage = "D:/mingw_installation/home/hurtbadly/Downloads/pet_orig_pat_3_sudy_0.nii.gz"
newImage = "D:/mingw_installation/home/hurtbadly/Downloads/volume-0.nii.gz"
strangeSpacingImage = "D:/mingw_installation/home/hurtbadly/Downloads/Output Volume_1.nii.gz"
extremeTestImage = "D:/mingw_installation/home/hurtbadly/Downloads/extreme_test_one.nii.gz"
bmaNiftiImage = "D:/mingw_installation/home/hurtbadly/Downloads/bma.nii.gz"
h5File = "D:/mingw_installation/home/hurtbadly/Downloads/locc.h5"
h5NiftiImage = "D:/mingw_installation/home/hurtbadly/Downloads/hdf5.nii.gz"

"""
NOTE : only one type of modality at a time in multi-image is supported.
"""



# medEyeStruct = MedEye3d.SegmentationDisplay.displayImage([[ctNiftiImage], [ctNiftiImage]]) #multi image displays
# medEyeStruct = MedEye3d.SegmentationDisplay.displayImage([ctNiftiImage]) #singleImageDisplay
# imm, res, line_indices = MedEye3d.ShaderAndVerticiesForSuperVoxels.get_example_sv_to_render()
# @info "Slice : 41 , Axis : 3 , Plane : Transversal"
# @info imm
# @info res
# @info line_indices


supervoxelDict = MedEye3d.ShadersAndVerticiesForSupervoxels.get_example_sv_to_render(h5File, "tetr_dat")

# @info typeof(supervoxel_vertices) #Vector{FLoat32}
# @info typeof(supervoxel_indices) #Vector{UInt32}

medEyeStruct = MedEye3d.SegmentationDisplay.displayImage(h5NiftiImage, svVertAndInd=supervoxelDict)
# #for SIngle you are strictly only supposed to pass it like : [image_ct, imagep]

# displayData = MedEye3d.DisplayDataManag.getDisplayedData(medEyeStruct, [Int32(1)]) #passing the active texture number


# #we need to check if the return type of the displayData is a single Array{Float32,3} or a vector{Array{Float32,3}}
# # now in this case we are setting random noise over the manualModif Texture voxel layer, and the manualModif texture defaults to 2 for active number
# fb = h5open(h5File, "r")

# displayData[1][1:128, 1:128, 1:128] = fb["im"][:, :, :]
# # displayData[2][:, :, :] = randn(Float32, size(displayData[2]))


# # @info "look here" typeof(displayData)
# MedEye3d.DisplayDataManag.setDisplayedData(medEyeStruct, displayData)


