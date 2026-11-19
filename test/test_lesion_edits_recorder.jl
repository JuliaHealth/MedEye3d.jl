using Test
using HDF5
using MedEye3d

const MEH = MedEye3d.SegmentationDisplay.MakieEventHandlers
const LER = MedEye3d.LesionEditsRecorder

@testset "Heat-GDT Speed, Steps/Frame & Inverted Size Controls" begin
    # 1. Speed interval: lower speed -> longer interval (up to 1.5s)
    orig_spd = MEH.HEAT_DISPLAY_RAMP_RATE[]
    orig_st = MEH.heatgdt_steps_per_frame[]
    try
        MEH.HEAT_DISPLAY_RAMP_RATE[] = 0.2f0
        @test MEH._current_heat_tick_interval() ≈ 1.50 atol=1e-4

        MEH.HEAT_DISPLAY_RAMP_RATE[] = 1.0f0
        @test MEH._current_heat_tick_interval() ≈ 0.60 atol=1e-4

        MEH.HEAT_DISPLAY_RAMP_RATE[] = 12.0f0
        @test MEH._current_heat_tick_interval() ≈ 0.05 atol=1e-4

        MEH.HEAT_DISPLAY_RAMP_RATE[] = 25.0f0
        @test MEH._current_heat_tick_interval() ≈ 0.024 atol=1e-4

        MEH.heatgdt_steps_per_frame[] = 2
        @test MEH.heatgdt_steps_per_frame[] == 2
    finally
        MEH.HEAT_DISPLAY_RAMP_RATE[] = orig_spd
        MEH.heatgdt_steps_per_frame[] = orig_st
    end

    # 2. Inverted Size ±: negative = smaller (higher theta up to 0.98), positive = larger (lower theta)
    base_theta = 0.001f0
    @test MEH._compute_posthoc_theta(base_theta, 0.0) ≈ base_theta atol=1e-6
    @test MEH._compute_posthoc_theta(base_theta, -10.0) ≈ 0.98f0 atol=1e-4
    @test MEH._compute_posthoc_theta(base_theta, -6.0) ≈ 0.1f0 atol=1e-4
    @test MEH._compute_posthoc_theta(base_theta, 6.0) ≈ 1.0f-5 atol=1e-7
    @test MEH._compute_posthoc_theta(base_theta, -5.0) > MEH._compute_posthoc_theta(base_theta, 0.0)
    @test MEH._compute_posthoc_theta(base_theta, 5.0) < MEH._compute_posthoc_theta(base_theta, 0.0)
end

@testset "LesionEditsRecorder Separate HDF5, 64x64x64 Patches & Connected Elongation" begin
    tmp_dir = mktempdir()
    src_h5 = joinpath(tmp_dir, "preprocessed_volumes.h5")
    # Create a dummy main HDF5 to verify it is NEVER modified
    HDF5.h5open(src_h5, "w") do f
        f["dummy"] = Int32[1, 2, 3]
    end
    src_mtime = stat(src_h5).mtime

    edits_h5 = LER.set_case_h5_path!(src_h5; spacing=(1.0, 1.0, 2.0), origin=(-50.0, -50.0, -100.0))
    @test edits_h5 == joinpath(tmp_dir, "lesion_edits.h5")

    W, H, D = 160, 160, 120
    ct_vol = Int16.(fill(-500, W, H, D))
    pet_vol = Float32.(zeros(W, H, D))
    seg_vol = zeros(Int16, W, H, D)

    # 1. Initial Heat-GDT lesion 1 at (40, 40, 40)
    seg_vol[38:43, 38:42, 38:42] .= Int16(1)
    uid1 = LER.record_heatgdt_event!(0, 1, (40, 40, 40), seg_vol; theta=0.001f0, K=20)
    @test uid1 !== nothing

    # 2. Elongated continuation click at (75, 40, 40) connected to (40, 40, 40)
    # Note: (75, 40, 40) is outside primary patch [9:72, 9:72, 9:72], so an overflow subpatch must be created!
    seg_vol[43:78, 39:41, 39:41] .= Int16(1)
    uid1_cont = LER.record_heatgdt_event!(0, 1, (75, 40, 40), seg_vol; theta=0.001f0, K=20)
    @test uid1_cont == uid1

    # 3. Manual paint & manual erase on lesion 1
    seg_vol[45, 45, 40] = Int16(1)
    n_paint = LER.record_brush_voxels!(0, 1, [(45, 45, 40)], false, seg_vol; brush_radius=1)
    @test n_paint == 1

    seg_vol[38, 38, 38] = Int16(0)
    n_erase = LER.record_brush_voxels!(0, 1, [(38, 38, 38)], true, seg_vol; brush_radius=1)
    @test n_erase == 1

    # 4. Post-hoc size adjustment on lesion 1
    uid_ph = LER.record_posthoc_event!(0, 1, "posthoc_size", seg_vol; theta=0.05f0, dilation=0, smoothing=0.0f0)
    @test uid_ph == uid1

    # 5. Disconnected lesion with same ID=1 far away at (130, 130, 90) -> must create a NEW lesion group
    seg_vol[128:132, 128:132, 88:92] .= Int16(1)
    uid2 = LER.record_heatgdt_event!(0, 1, (130, 130, 90), seg_vol; theta=0.001f0, K=20)
    @test uid2 !== nothing
    @test uid2 != uid1

    # 6. Flush asynchronously to independent lesion_edits.h5
    @test LER.has_pending_edits()
    t_flush = LER.flush_all_edits_async!(ct_vol=ct_vol, pet_vol=pet_vol)
    @test t_flush isa Task
    @test fetch(t_flush) == true
    @test !LER.has_pending_edits()
    @test isfile(edits_h5)
    @test stat(src_h5).mtime == src_mtime  # Main HDF5 untouched!

    HDF5.h5open(edits_h5, "r") do f
        @test haskey(f, "CT/volume")
        @test haskey(f, "PET/volume")
        @test size(f["CT/volume"]) == (W, H, D)
        @test size(f["PET/volume"]) == (W, H, D)
        @test haskey(f, "lesions/$uid1")
        @test haskey(f, "lesions/$uid2")

        g1 = f["lesions/$uid1"]
        @test size(g1["initial_heat_mask"]) == (64, 64, 64)
        @test size(g1["final_mask"]) == (64, 64, 64)
        @test size(g1["manual_painted_mask"]) == (64, 64, 64)
        @test size(g1["manual_erased_mask"]) == (64, 64, 64)
        @test sum(read(g1["manual_painted_mask"])) == 1
        @test sum(read(g1["manual_erased_mask"])) == 1
        @test length(attrs(g1)["chunk_center_vox"]) == 3
        @test length(attrs(g1)["chunk_center_mm"]) == 3
        @test haskey(g1, "subpatches/subpatch_1/final_mask")
        @test size(g1["subpatches/subpatch_1/final_mask"]) == (64, 64, 64)
        @test haskey(g1, "events/0001_heat_gdt")
        @test haskey(g1, "events/0002_heat_gdt")
        @test haskey(g1, "events/0003_manual_paint")
        @test haskey(g1, "events/0004_manual_erase")
        @test haskey(g1, "events/0005_posthoc_size")
    end

    LER.reset_recorder!()
    rm(tmp_dir; recursive=true, force=true)
end

@testset "XNAT Save/Load Wiring for Main + Lesion Editing HDF5" begin
    XC = MedEye3d.XnatClient
    @test isdefined(XC, :upload_preprocessed_hdf5)
    @test isdefined(XC, :upload_lesion_edits_hdf5)
    @test isdefined(XC, :download_preprocessed_hdf5)
    @test isdefined(XC, :download_lesion_edits_hdf5)

    # Filename predicates must cleanly separate main preprocessed HDF5 from lesion_edits HDF5
    @test XC._is_preprocessed_h5_filename("Pat22_preprocessed.h5") == true
    @test XC._is_preprocessed_h5_filename("preprocessed_volumes.h5") == true
    @test XC._is_preprocessed_h5_filename("Pat22_lesion_edits.h5") == false
    @test XC._is_preprocessed_h5_filename("lesion_edits.h5") == false
    @test XC._is_lesion_edits_h5_filename("Pat22_lesion_edits.h5") == true
    @test XC._is_lesion_edits_h5_filename("lesion_edits.h5") == true
    @test XC._is_lesion_edits_h5_filename("Pat22_preprocessed.h5") == false

    lmw_src = read(joinpath(@__DIR__, "..", "src", "display", "LesionMetadataWindow.jl"), String)
    @test occursin("XnatClient.upload_preprocessed_hdf5(creds, proj, subj, h5_active)", lmw_src)
    @test occursin("XnatClient.upload_lesion_edits_hdf5(creds, proj, subj, edits_h5)", lmw_src)
    @test occursin("if ok_annot && ok_h5 && ok_edits", lmw_src)
    @test occursin("XnatClient.download_lesion_edits_hdf5(creds, proj, subj, edits_dest", lmw_src)
    @test isdefined(LER, :flush_all_edits_async!)
    @test isdefined(MEH, :flush_all_dirty_masks_async!)
    @test MEH.heatgdt_edge_modality[] === :ct_only
    app_src = read(joinpath(@__DIR__, "..", "src", "packaging", "AppMain.jl"), String)
    @test occursin("init_mod === :ct_only ? \"diffusivity_ct\"", app_src)
    @test occursin("sw_mod === :ct_only ? \"diffusivity_ct\"", app_src)
end

@testset "HDF5 & XNAT Manual Modification Status + User Tracking" begin
    LMW = MedEye3d.LesionMetadataWindow
    tmp_dir = mktempdir()
    h5p = joinpath(tmp_dir, "test_mod.h5")

    # 1. Unmodified HDF5
    HDF5.h5open(h5p, "w") do f
        mg = HDF5.create_group(f, "_meta_")
        LMW.write_h5_modification_status!(mg, false, "")
    end
    is_mod, user, ts = LMW.read_h5_modification_status(h5p)
    @test is_mod == false
    @test user == ""
    @test ts == ""

    # 2. Manually modified HDF5 records user + timestamp
    HDF5.h5open(h5p, "r+") do f
        LMW.write_h5_modification_status!(f["_meta_"], true, "dr_kowalski"; ts="2026-11-17T16:10:00")
    end
    is_mod2, user2, ts2 = LMW.read_h5_modification_status(h5p)
    @test is_mod2 == true
    @test user2 == "dr_kowalski"
    @test ts2 == "2026-11-17T16:10:00"

    # 3. Verify XNAT status_dict & GUI status text wiring
    lmw_src = read(joinpath(@__DIR__, "..", "src", "display", "LesionMetadataWindow.jl"), String)
    @test occursin("\"manually_modified\" => is_mod", lmw_src)
    @test occursin("\"modified_by\" => is_mod ? mod_user : \"\"", lmw_src)
    @test occursin("Status: PROCESSED [Modified by \$", lmw_src)
    @test occursin("Status: PREPROCESSED [Unmodified]", lmw_src)

    rm(tmp_dir; recursive=true, force=true)
end
