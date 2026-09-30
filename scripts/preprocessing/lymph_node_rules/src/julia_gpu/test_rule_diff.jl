using JSON
using MedImages
using CUDA

include("main/DagOrchestrator.jl")
using .DagOrchestrator

h5_path = "data/processed_cases_restored/FDM_DPI-2024-7-KRN_Prostata_bimodal__PETPSMA_0__Pat44/primary_masks.h5"

rules_to_test = [
    "Abdominal_Obturator_left",
    "Abdominal_Obturator_right",
    "Abdominal_Presacral",
    "Abdominal_Internal_Iliac_Left",
    "Abdominal_Internal_Iliac_Right",
    "Abdominal_Common_Iliac_Left",
    "Abdominal_Common_Iliac_Right"
]

println("Testing rules individually...")
