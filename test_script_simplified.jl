using MedEye3d

#Data downloaded from the following google drive links :

# https://drive.google.com/file/d/1Segr6BC_ma9bKNmM8lzUBaJLQChkAWKA/view?usp=drive_link
# https://drive.google.com/file/d/1PqHTQXOVTWx0FQSDE0oH4hRKNCE9jCx8/view?usp=drive_link
#modify paths to your downloaded data accordingly

ctNiftiImage = "/home/hurtbadly/Downloads/ct_soft_pat_3_sudy_0.nii.gz"
petNiftiImage = "/home/hurtbadly/Downloads/pet_orig_pat_3_sudy_0.nii.gz"
medImageObjects = MedEye3d.SegmentationDisplay.loadRegisteredImages([ctNiftiImage])

@info medImageObjects[1].image_type


MedEye3d.SegmentationDisplay.summonVisualizer(medImageObjects)
