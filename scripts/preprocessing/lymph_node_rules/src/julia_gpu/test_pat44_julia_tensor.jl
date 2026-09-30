using HDF5
h5open("data/processed_cases_restored/FDM_DPI-2024-7-KRN_Prostata_bimodal__PETPSMA_0__Pat44/primary_masks.h5", "r") do h5
    println("Direction: ", read(h5["metadata/direction"]))
end
