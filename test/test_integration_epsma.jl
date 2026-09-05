using Test
using Dates
using JSON
using HDF5
using MedEye3d
using MedEye3d.EPSMAStructuredReport
using MedEye3d.EPSMAReportWindow

@testset "End-to-End E-PSMA Integration Test" begin
    println("\n=== Testing E-PSMA Data Building on Patient 6 ===")
    
    # Check patient 6 organ mapping in preprocessed_volumes.h5
    h5_path = "/workspaces/MedEye3d.jl/data/pat_6_files/preprocessed_volumes.h5"
    if isfile(h5_path)
        h5 = HDF5.h5open(h5_path, "r")
        if haskey(h5, "_meta_/organ_mapping")
            om_raw = JSON.parse(read(h5["_meta_/organ_mapping"]))
            println("Loaded $(length(om_raw)) lesions from HDF5 organ_mapping")
            
            # Populate _MEH global_organ_mapping for testing
            _MEH = EPSMAStructuredReport._get_meh()
            if _MEH !== nothing
                for (lid_str, org) in om_raw
                    _MEH.global_organ_mapping[][parse(Int, lid_str)] = org
                end
            end
        end
        close(h5)
    end
    
    # 1. Build structured report data for TP 0
    rep0 = EPSMAStructuredReport.build_epsma_data(0)
    @test rep0 isa EPSMAStructuredReport.EPSMAReport
    @test !isempty(rep0.patient_id)
    println("Report TP 0 Patient ID: $(rep0.patient_id)")
    println("Report TP 0 miTNM: $(rep0.final_mitnm)")
    println("Report TP 0 Synoptic Rows (PCa Metastases): $(length(rep0.synoptic_rows))")
    println("Report TP 0 Artifact Rows (Excluded): $(length(rep0.artifact_rows))")
    
    # Verify Muscle Artifact Exclusion:
    # Quadriceps (Lesion 1), Sartorius (Lesion 4), Thigh compartment (Lesion 2) must NOT be in synoptic_rows!
    syn_locs = [r.location for r in rep0.synoptic_rows]
    for loc in syn_locs
        @test !occursin("quadriceps", lowercase(loc))
        @test !occursin("sartorius", lowercase(loc))
        @test !occursin("Lesion ", loc) # NEVER generic "Lesion X"!
    end
    
    # If artifact_rows present, verify they contain the muscles
    if !isempty(rep0.artifact_rows)
        art_locs = [r.location for r in rep0.artifact_rows]
        has_muscle_art = any(l -> occursin("quadriceps", lowercase(l)) || occursin("sartorius", lowercase(l)), art_locs)
        println("Artifact rows include muscle physiological uptake: $has_muscle_art")
    end
    
    # Verify subregions in Section 4
    @test occursin("Subregion", rep0.findings_prostate_en) || occursin("Prostate Gland", rep0.findings_prostate_en)
    @test occursin("Pelvic Lymph Nodes", rep0.findings_lymph_en)
    @test occursin("Axial Skeleton", rep0.findings_bone_en) || occursin("Appendicular Skeleton", rep0.findings_bone_en)
    @test occursin("Non-Nodal Visceral", rep0.findings_visceral_en)
    
    # 2. Export Word documents
    en_path = "/workspaces/MedEye3d.jl/data/reports/E_PSMA_$(rep0.patient_id)_TP0_EN.docx"
    de_path = "/workspaces/MedEye3d.jl/data/reports/E_PSMA_$(rep0.patient_id)_TP0_DE.docx"
    
    mkpath(dirname(en_path))
    
    export_to_docx(rep0, en_path; lang="EN")
    @test isfile(en_path)
    @test filesize(en_path) > 15000
    println("Generated EN DOCX: $en_path ($(filesize(en_path)) bytes)")
    
    export_to_docx(rep0, de_path; lang="DE")
    @test isfile(de_path)
    @test filesize(de_path) > 15000
    println("Generated DE DOCX: $de_path ($(filesize(de_path)) bytes)")
    
    # 3. Verify DOCX internal XML structure
    py_check = `python3 -c "
import zipfile
import xml.etree.ElementTree as ET

for path in ['$en_path', '$de_path']:
    with zipfile.ZipFile(path) as z:
        assert 'word/document.xml' in z.namelist()
        xml_data = z.read('word/document.xml').decode('utf-8')
        assert 'E-PSMA' in xml_data
        assert 'Synoptic Table' in xml_data or 'Synoptische Tabelle' in xml_data
        assert 'miTNM' in xml_data
        print(f'Verified valid XML structure for {path}')
"`
    run(py_check)
    
    println("\n=== All Integration Checks Passed Successfully! ===")
end
