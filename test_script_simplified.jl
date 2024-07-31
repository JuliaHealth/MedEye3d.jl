using MedEye3d

dirOfExampleCT = "/home/hurtbadly/Downloads/ct_soft_pat_3_sudy_0.nii.gz"
dirOfExamplePET = "/home/hurtbadly/Downloads/pet_orig_pat_3_sudy_0.nii.gz"
medImageObject, voxelData, spacing = MedEye3d.SegmentationDisplay.loadRegisteredImages(dirOfExamplePET)

studyOptions = ["CT", "PET"]
MedEye3d.SegmentationDisplay.summonVisualizer(medImageObject, voxelData, spacing, studyOptions[2])
