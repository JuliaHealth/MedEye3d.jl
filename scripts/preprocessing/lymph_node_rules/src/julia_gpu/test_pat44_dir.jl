using NIfTI
case_dir = "data/processed_cases_restored/FDM_DPI-2024-7-KRN_Prostata_bimodal__PETPSMA_0__Pat44"
nii = niread(joinpath(case_dir, "segmentations/autochthon_left.nii.gz"))
println(nii.header.pixdim[2:4])
println("QFORM:")
println(nii.header.qform_code)
println(nii.header.quatern_b, " ", nii.header.quatern_c, " ", nii.header.quatern_d)
println("SFORM:")
println(nii.header.sform_code)
println(nii.header.srow_x)
println(nii.header.srow_y)
println(nii.header.srow_z)
