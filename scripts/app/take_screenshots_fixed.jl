using MedEye3d
using GLMakie
using HDF5
using JSON

function main()
    active = Observable("1")
    ids = Observable(["1", "2", "3 [SWEEP]"]) 
    
    res = MedEye3d.LesionMetadataWindow.create_metadata_window(active, ids, nothing)
    fig_meta = res.fig
    
    # Wait for initial async load
    for i in 1:50
        if !isempty(MedEye3d.LesionMetadataWindow._active_lesion_db[][])
            break
        end
        sleep(0.1)
    end
    sleep(0.5)
    
    # Iterate through all Menu and Button elements to FORCE them manually
    for block in fig_meta.content
        if block isa Menu
            opts = block.options[]
            if "POST_RLT" in opts
                idx = findfirst(==("POST_RLT"), opts)
                block.i_selected[] = idx
            elseif "CORRECTED" in opts
                idx = findfirst(==("CORRECTED"), opts)
                block.i_selected[] = idx
            end
        elseif block isa Button
            if block.label[] == "☆ Key Image"
                block.label[] = "★ Key Image"
                block.buttoncolor[] = RGBf(0.9, 0.8, 0.2)
            end
        end
    end
    
    sleep(0.5)
    GLMakie.save("/workspaces/MedEye3d.jl/data/screenshots/clinical_metadata_panel_keyimage.png", fig_meta)
    
    # Show Sweep candidate
    active[] = "3 [SWEEP]"; sleep(0.5)
    GLMakie.save("/workspaces/MedEye3d.jl/data/screenshots/clinical_metadata_panel_sweep.png", fig_meta)
    
    # EPSMA Report
    report = MedEye3d.EPSMAStructuredReport.EPSMAReport(
        patient_id = "PAT_TEST_123",
        clinical_profile = "POST_RLT",
        final_mitnm = "miT0 miN0 miM1b",
        overall_recip = "PARTIAL METABOLIC RESPONSE (PMR / RECIP-PR)",
        synoptic_rows = [
            MedEye3d.EPSMAStructuredReport.EPSMALesionRow(
                id = 1, location = "Right Iliac Bone", mitnm = "miM1b", size_str = "12 mm",
                psma_q = "SUVmax 15.2", psma_v_num = 3, is_key_image = true, is_new = false, recip_status = "RECIP-PR"
            ),
            MedEye3d.EPSMAStructuredReport.EPSMALesionRow(
                id = 2, location = "Prostate Gland", mitnm = "miT0", size_str = "0 mm",
                psma_q = "SUVmax 2.1", psma_v_num = 0, is_key_image = false, is_new = false, recip_status = "RECIP-CR"
            )
        ]
    )
    
    MedEye3d.EPSMAReportWindow.open_epsma_report_window(report)
    fig_report = MedEye3d.EPSMAReportWindow.active_report_fig[]
    
    sleep(1.0)
    GLMakie.save("/workspaces/MedEye3d.jl/data/screenshots/clinical_epsma_report_top.png", fig_report)
    
    # Scroll down to bottom
    for _ in 1:20
        fig_report.scene.events.scroll[] = (0.0, -10.0)
        sleep(0.05)
    end
    
    sleep(1.0)
    GLMakie.save("/workspaces/MedEye3d.jl/data/screenshots/clinical_epsma_report_bottom.png", fig_report)
    
    println("Fixed screenshots generated")
end

main()
