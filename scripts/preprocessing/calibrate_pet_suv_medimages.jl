#!/usr/bin/env julia
"""
calibrate_pet_suv_medimages.jl

Uses MedImages.jl (MedImages.calculate_suv_factor) to calculate exact decay-corrected
SUV factors from DICOM metadata and calibrate PET NIfTI volumes in:
  data/cases/psma_patient_all_tp/
  data/processed/Patient_1_277820/PET/
  data/processed/Patient_2_277735/PET/
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))

using MedImages
using NIfTI
using JSON
using Statistics

println("============================================================")
println("  PET SUV Calibration via MedImages.jl")
println("============================================================")

raw_dicom_root = joinpath(@__DIR__, "..", "..", "data", "raw_dicom")
cases_root = joinpath(@__DIR__, "..", "..", "data", "cases", "psma_patient_all_tp")
proc_root = joinpath(@__DIR__, "..", "..", "data", "processed")
meta_file = joinpath(raw_dicom_root, "pet_suv_metadata.json")

if !isfile(meta_file)
    error("Metadata file not found: $meta_file")
end

meta_json = JSON.parsefile(meta_file)

function get_suv_factor(key::String)
    if !haskey(meta_json, key)
        error("Key $key not found in $meta_file")
    end
    raw_dict = meta_json[key]
    
    # Construct MedImage.metadata dictionary
    meta = Dict{Any, Any}(
        "PatientWeight" => Float64(raw_dict["PatientWeight"]),
        "RadiopharmaceuticalInformationSequence" => [
            Dict{Any, Any}(
                "RadionuclideTotalDose" => Float64(raw_dict["RadiopharmaceuticalInformationSequence"][1]["RadionuclideTotalDose"]),
                "RadionuclideHalfLife" => Float64(raw_dict["RadiopharmaceuticalInformationSequence"][1]["RadionuclideHalfLife"]),
                "RadiopharmaceuticalStartTime" => string(raw_dict["RadiopharmaceuticalInformationSequence"][1]["RadiopharmaceuticalStartTime"])
            )
        ],
        "AcquisitionTime" => string(raw_dict["AcquisitionTime"]),
        "SeriesDescription" => string(raw_dict["SeriesDescription"])
    )
    
    dummy_vox = fill(1.0f0, 2, 2, 2)
    mi = MedImage(
        voxel_data = dummy_vox,
        origin = (0.0, 0.0, 0.0),
        spacing = (1.0, 1.0, 1.0),
        direction = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        patient_id = "pat",
        image_type = MedImages.MedImage_data_struct.PET_type,
        image_subtype = MedImages.MedImage_data_struct.PSMA_subtype,
        legacy_file_name = "pet.nii.gz",
        metadata = meta
    )
    
    factor = MedImages.calculate_suv_factor(mi)
    return factor
end

fac_tp0 = get_suv_factor("pat_2_PET WB")
fac_tp1 = get_suv_factor("pat_2_PET WB spaet")
fac_tp2 = get_suv_factor("pat_1_PET WB spaet")

println("\nMedImages.jl Calculated SUV Factors:")
println("  TP 0 (Patient 2 Early, 'PET WB'):       ", fac_tp0)
println("  TP 1 (Patient 2 Late,  'PET WB spaet'): ", fac_tp1)
println("  TP 2 (Patient 1 Late,  'PET WB spaet'): ", fac_tp2)

function calibrate_nifti(in_path::String, out_path::String, factor::Float64)
    if !isfile(in_path)
        @warn "File not found: $in_path"
        return
    end
    nii = niread(in_path)
    raw_vals = Float32.(nii.raw)
    suv_vals = raw_vals .* Float32(factor)
    
    out_nii = NIVolume(nii.header, nii.extensions, suv_vals)
    niwrite(out_path, out_nii)
    
    pos = suv_vals[suv_vals .> 0]
    p50 = isempty(pos) ? 0.0 : round(quantile(pos, 0.50), digits=2)
    p95 = isempty(pos) ? 0.0 : round(quantile(pos, 0.95), digits=2)
    p99 = isempty(pos) ? 0.0 : round(quantile(pos, 0.99), digits=2)
    mx = isempty(pos) ? 0.0 : round(maximum(pos), digits=2)
    println("  Calibrated $(basename(out_path)): min=$(round(minimum(suv_vals), digits=2)), p50=$p50, p95=$p95, p99=$p99, max=$mx SUV")
end

raw_patients_root = joinpath(@__DIR__, "..", "..", "data", "patients")

println("\nCalibrating files from raw master copies (data/patients/)...")
calibrate_nifti(
    joinpath(raw_patients_root, "Patient_2_277735", "PET", "pet_wb.nii.gz"),
    joinpath(cases_root, "SUV_PET_Image_0.nii.gz"),
    fac_tp0
)
calibrate_nifti(
    joinpath(raw_patients_root, "Patient_2_277735", "PET", "pet_wb_spaet.nii.gz"),
    joinpath(cases_root, "SUV_PET_Image_1.nii.gz"),
    fac_tp1
)
calibrate_nifti(
    joinpath(raw_patients_root, "Patient_1_277820", "PET", "pet_wb_spaet.nii.gz"),
    joinpath(cases_root, "SUV_PET_Image_2.nii.gz"),
    fac_tp2
)

println("\nUpdating files in data/processed/ archives...")
calibrate_nifti(
    joinpath(raw_patients_root, "Patient_2_277735", "PET", "pet_wb.nii.gz"),
    joinpath(proc_root, "Patient_2_277735", "PET", "pet_wb.nii.gz"),
    fac_tp0
)
calibrate_nifti(
    joinpath(raw_patients_root, "Patient_2_277735", "PET", "pet_wb_spaet.nii.gz"),
    joinpath(proc_root, "Patient_2_277735", "PET", "pet_wb_spaet.nii.gz"),
    fac_tp1
)
calibrate_nifti(
    joinpath(raw_patients_root, "Patient_1_277820", "PET", "pet_wb_spaet.nii.gz"),
    joinpath(proc_root, "Patient_1_277820", "PET", "pet_wb_spaet.nii.gz"),
    fac_tp2
)

println("\nPET SUV calibration successfully completed via MedImages.jl!")
