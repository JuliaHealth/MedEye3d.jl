using MedEye3d

#Data downloaded from the following google drive links :

# https://drive.google.com/file/d/1Segr6BC_ma9bKNmM8lzUBaJLQChkAWKA/view?usp=drive_link
# https://drive.google.com/file/d/1PqHTQXOVTWx0FQSDE0oH4hRKNCE9jCx8/view?usp=drive_link
#modify paths to your downloaded data accordingly

ctNiftiImage = "D:/mingw_installation/home/hurtbadly/Downloads/ct_soft_pat_3_sudy_0.nii.gz"
petNiftiImage = "D:/mingw_installation/home/hurtbadly/Downloads/pet_orig_pat_3_sudy_0.nii.gz"
newImage = "D:/mingw_installation/home/hurtbadly/Downloads/volume-0.nii.gz"
strangeSpacingImage = "D:/mingw_installation/home/hurtbadly/Downloads/Output Volume_1.nii.gz"
extremeTestImage = "D:/mingw_installation/home/hurtbadly/Downloads/extreme_test_one.nii.gz"
channelStruct = MedEye3d.SegmentationDisplay.displayImage(extremeTestImage)
mainChannel = channelStruct.channel
