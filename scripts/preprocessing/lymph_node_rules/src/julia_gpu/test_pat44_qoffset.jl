using NIfTI
case_dir = "data/processed_cases_restored/FDM_DPI-2024-7-KRN_Prostata_bimodal__PETPSMA_0__Pat44"
nii = niread(joinpath(case_dir, "segmentations/autochthon_left.nii.gz"))
println(nii.header.qoffset_x, " ", nii.header.qoffset_y, " ", nii.header.qoffset_z)
