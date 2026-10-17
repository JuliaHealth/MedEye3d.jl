using Test
using HDF5
using JSON
using MedEye3d

const MEH = MedEye3d.SegmentationDisplay.MakieEventHandlers
const LMW = MedEye3d.LesionMetadataWindow
const LA  = MedEye3d.LesionAssociation
const LT  = MedEye3d.LesionTracker
const Meas = MedEye3d.Measurements
const MEv = MedEye3d.MakieEvents

@testset "All Modifications Saved to HDF5 (Not NIfTIs)" begin
    tmp_dir = mktempdir()
    h5_file = joinpath(tmp_dir, "preprocessed_volumes.h5")
    json_file = joinpath(tmp_dir, "test_annotations.json")

    # ── 0. Create realistic 2-timepoint HDF5 dataset (BASELINE + TFM_1) ──────
    HDF5.h5open(h5_file, "w") do f
        g0 = create_group(f, "BASELINE")
        HDF5.attributes(g0)["modality"] = "PET"
        HDF5.attributes(g0)["node_name"] = "PET_Lesions_0"
        g0["Fixed_CT_Volume_0"] = zeros(Float32, 20, 20, 20)
        pet0 = fill(1.5f0, 20, 20, 20)
        pet0[5:8, 5:8, 5:8] .= 12.0f0
        g0["SUV_PET_Image_0"] = pet0
        mask0 = zeros(Int16, 20, 20, 20)
        mask0[5:7, 5:7, 5:7] .= Int16(1)
        mask0[12:14, 12:14, 12:14] .= Int16(2)
        g0["PET_Lesions_0"] = mask0

        g1 = create_group(f, "TFM_1")
        HDF5.attributes(g1)["modality"] = "PET"
        HDF5.attributes(g1)["node_name"] = "PET_Lesions_1"
        g1["Fixed_CT_Volume_1"] = zeros(Float32, 20, 20, 20)
        pet1 = fill(1.2f0, 20, 20, 20)
        pet1[6:9, 6:9, 6:9] .= 9.5f0
        g1["SUV_PET_Image_1"] = pet1
        mask1 = zeros(Int16, 20, 20, 20)
        mask1[6:8, 6:8, 6:8] .= Int16(1)
        g1["PET_Lesions_1"] = mask1

        c_grp = create_group(f, "CENTROIDS")
        c_grp["tp0_lid1"] = Int32[6, 6, 6]
        c_grp["tp0_lid2"] = Int32[13, 13, 13]
        c_grp["tp1_lid1"] = Int32[7, 7, 7]

        b_grp = create_group(f, "BONE_SUBSEG")
        b_grp["tp_0_lesion_1_surf"] = Int64[1]
        b_grp["tp_0_lesion_1_marr"] = Int64[2]

        m_grp = create_group(f, "_meta_")
        m_grp["preflipped"] = Int32(1)
        m_grp["organ_mapping"] = JSON.json(Dict("0" => Dict("1" => "ilium", "2" => "liver"), "1" => Dict("1" => "ilium")))
        m_grp["segment_names.json"] = JSON.json(Dict("0" => Dict("1" => "Left Ilium", "2" => "Liver"), "1" => Dict("1" => "Left Ilium")))
        m_grp["matches.json"] = JSON.json([
            Dict(
                "name" => "PET_Lesions_0",
                "raw_lesion" => "Segment_1",
                "node" => "parent",
                "lesion" => "Segment_1",
                "volume_mm3" => 100.0,
                "group_id" => 1,
                "children" => [
                    Dict(
                        "name" => "PET_Lesions_1",
                        "raw_lesion" => "Segment_1",
                        "lesion" => "Segment_1",
                        "volume_mm3" => 100.0,
                        "group_id" => 1
                    )
                ]
            )
        ])
    end

    # Configure MEH state to point to our test HDF5
    MEH.h5_path_ref[] = h5_file
    MEH.h5_save_path_ref[] = h5_file
    MEH.studies_ref[] = [
        ("PET", 0, "20250101", "Fixed_CT_Volume_0", "SUV_PET_Image_0", "PET_Lesions_0", "PET_Lesions_0", ""),
        ("PET", 1, "20250601", "Fixed_CT_Volume_1", "SUV_PET_Image_1", "PET_Lesions_1", "PET_Lesions_1", "1")
    ]
    MEH.tp_node_names[0] = "PET_Lesions_0"
    MEH.tp_node_names[1] = "PET_Lesions_1"
    MEH.current_tp_index[] = 0
    MEH.compare_right_tp[] = 1
    empty!(MEH.tp_data_cache)
    empty!(MEH.dirty_mask_tps)
    empty!(MEH.dirty_bone_subsegs)
    empty!(MEH.bone_subsegments_cache)
    empty!(MEH._ai_mask_backups)
    empty!(MEH.lesion_centroids_cache)

    # Populate tp_data_cache for TP0 and TP1 using MEH.TpCacheEntry
    HDF5.h5open(h5_file, "r") do f
        m0_i16 = read(f["BASELINE/PET_Lesions_0"])
        m0_i8  = Int8.(clamp.(m0_i16, -128, 127))
        p0 = read(f["BASELINE/SUV_PET_Image_0"])
        c0 = read(f["BASELINE/Fixed_CT_Volume_0"])
        MEH.tp_data_cache[0] = MEH.TpCacheEntry(
            c0, p0, m0_i8, zeros(Int8, 20, 20, 20), nothing, m0_i16, nothing
        )
        MEH._ai_mask_backups[(0, 1)] = [CartesianIndex(x, y, z) for x in 5:7 for y in 5:7 for z in 5:7]

        m1_i16 = read(f["TFM_1/PET_Lesions_1"])
        m1_i8  = Int8.(clamp.(m1_i16, -128, 127))
        p1 = read(f["TFM_1/SUV_PET_Image_1"])
        c1 = read(f["TFM_1/Fixed_CT_Volume_1"])
        MEH.tp_data_cache[1] = MEH.TpCacheEntry(
            c1, p1, m1_i8, zeros(Int8, 20, 20, 20), nothing, m1_i16, nothing
        )
    end

    # ── 1. Mask Painting & Erasing (Axial, Sagittal, Coronal + Compare TP1) ──
    @testset "1. Mask Edits (Axial, Sagittal, Coronal, Compare TP) Saved to HDF5" begin
        e0 = MEH.tp_data_cache[0]
        # Simulate Axial edit (z=10), Sagittal edit (x=15), Coronal edit (y=16) for Lesion 1
        e0.mask[8, 8, 10] = Int8(1)
        e0.mask_i16[8, 8, 10] = Int16(1)
        e0.mask[15, 4, 4] = Int8(1)
        e0.mask_i16[15, 4, 4] = Int16(1)
        e0.mask[3, 16, 4] = Int8(1)
        e0.mask_i16[3, 16, 4] = Int16(1)
        MEH.mark_tp_mask_dirty!(0)

        # Simulate Compare mode TP1 mask edit
        e1 = MEH.tp_data_cache[1]
        e1.mask[10, 10, 10] = Int8(1)
        e1.mask_i16[10, 10, 10] = Int16(1)
        MEH.mark_tp_mask_dirty!(1)

        MEH.flush_all_dirty_masks!()
        @test isempty(MEH.dirty_mask_tps)

        # Verify HDF5 contains _expert datasets and updated centroids, and NO .nii.gz was created
        @test isempty(filter(f -> endswith(f, ".nii") || endswith(f, ".nii.gz"), readdir(tmp_dir)))
        HDF5.h5open(h5_file, "r") do f
            @test haskey(f["BASELINE"], "PET_Lesions_0_expert")
            @test haskey(f["TFM_1"], "PET_Lesions_1_expert")
            exp0 = read(f["BASELINE/PET_Lesions_0_expert"])
            exp1 = read(f["TFM_1/PET_Lesions_1_expert"])
            @test exp0[8, 8, 10] == Int16(1)
            @test exp0[15, 4, 4] == Int16(1)
            @test exp0[3, 16, 4] == Int16(1)
            @test exp1[10, 10, 10] == Int16(1)
            @test haskey(f["CENTROIDS"], "tp0_lid1")
            @test haskey(f["CENTROIDS"], "tp1_lid1")
        end
    end

    # ── 2. Bone Subsegmentation (:surface / :marrow) Edits Saved to HDF5 ─────
    @testset "2. Bone Subsegmentation Manual Edits Saved to HDF5" begin
        surf_pts = [CartesianIndex(5, 5, 5), CartesianIndex(5, 6, 5)]
        marr_pts = [CartesianIndex(6, 6, 6), CartesianIndex(6, 7, 6)]
        MEH.bone_subsegments_cache[(0, 1)] = (surf_pts, marr_pts)
        MEH.mark_bone_subseg_dirty!(0, 1)

        MEH.flush_all_dirty_masks!()
        @test isempty(MEH.dirty_bone_subsegs)

        lin = LinearIndices((20, 20, 20))
        HDF5.h5open(h5_file, "r") do f
            @test haskey(f, "BONE_SUBSEG")
            bs = f["BONE_SUBSEG"]
            @test haskey(bs, "tp_0_lesion_1_surf")
            @test haskey(bs, "tp_0_lesion_1_marr")
            @test haskey(bs, "PET_Lesions_0_lesion_1_surf")
            @test haskey(bs, "lesion_1_surf")
            s_h5 = read(bs["tp_0_lesion_1_surf"])
            m_h5 = read(bs["tp_0_lesion_1_marr"])
            @test lin[CartesianIndex(5, 6, 5)] in s_h5
            @test lin[CartesianIndex(6, 7, 6)] in m_h5
        end
    end

    # ── 3. RevertToAIEvent Restores Original AI Voxels & Saves to HDF5 ───────
    @testset "3. RevertToAIEvent Restores Original AI Mask and Persists to HDF5" begin
        MEH.current_active_lesion_id[] = 1
        MEH.current_tp_index[] = 0
        # Before revert, voxel [8,8,10] is 1 from manual edit
        @test MEH.tp_data_cache[0].mask_i16[8, 8, 10] == Int16(1)
        MEH.reactToRevertToAI(MEv.RevertToAIEvent(1, 0), MEH.StateDataFields[])
        MEH.flush_all_dirty_masks!()

        # After revert, [8,8,10] is cleared back to 0 and original 5:7 cube is 1
        @test MEH.tp_data_cache[0].mask_i16[8, 8, 10] == Int16(0)
        @test count(==(Int16(1)), MEH.tp_data_cache[0].mask_i16) == 27
        HDF5.h5open(h5_file, "r") do f
            exp0 = read(f["BASELINE/PET_Lesions_0_expert"])
            @test exp0[8, 8, 10] == Int16(0)
            @test count(==(Int16(1)), exp0) == 27
        end
    end

    # ── 4. Measurements (Spheres & Lines) Saved to HDF5 ──────────────────────
    @testset "4. Sphere and Line Measurements Saved to HDF5" begin
        mock_obj = (
            measurements = [
                Meas.SphereMeasurement(id=1, center_idx=(6.0f0, 6.0f0, 6.0f0), radius_mm=8.5f0, suv_mean=5.2f0, suv_max=12.0f0, is_active=false)
            ],
            line_measurements = [
                Meas.LineMeasurement(id=1, start_idx=(2.0f0, 2.0f0, 6.0f0), end_idx=(10.0f0, 10.0f0, 6.0f0), length_mm=11.31f0, is_active=false)
            ]
        )
        ok = MEH.save_measurements_to_h5(0; obj=mock_obj)
        @test ok == true

        loaded_obj = (
            measurements = Meas.SphereMeasurement[],
            line_measurements = Meas.LineMeasurement[]
        )
        MEH.load_measurements_from_h5!(0, loaded_obj)
        @test length(loaded_obj.measurements) == 1
        @test loaded_obj.measurements[1].radius_mm ≈ 8.5f0
        @test loaded_obj.measurements[1].suv_max ≈ 12.0f0
        @test length(loaded_obj.line_measurements) == 1
        @test loaded_obj.line_measurements[1].length_mm ≈ 11.31f0
    end

    # ── 5. Metadata, Presets, Custom Options, Custom Fields, Display Config ──
    @testset "5. Metadata, Presets, Custom Options & Fields Saved to HDF5" begin
        # Custom preset
        LMW.add_metadata_preset!("Test HDF5 Preset", Dict(
            "BaseAnatomy" => "femur",
            "BaseAnatomySide" => "Right",
            "LesionType" => "Bone Meta",
            "Certainty" => "3",
            "NoCTCorrelate" => "false"
        ))
        # Custom option & custom field & display config
        LMW.add_global_custom_option("Radiotracer", "18F-FDG-CustomH5")
        LMW.save_global_custom_fields(Dict("TrialArm" => "Cohort-A"))
        LMW.save_display_config(Dict{String,Any}("label_opacity" => 0.75, "pet_ct_blend" => 0.35, "heatgdt_ramp_speed" => 4.0))

        # Lesion annotations
        db = Dict{String,Any}(
            "1" => Dict{String,Any}(
                "BaseAnatomy" => "femur",
                "BaseAnatomySide" => "Right",
                "LesionType" => "Bone Meta",
                "Certainty" => "3",
                "ObservationState" => "CORRECTED",
                "has_expert_edits" => true,
                "_display_name" => "1: Right Femur"
            )
        )
        LMW.save_annotations(db, json_file)

        # Verify all _meta_ datasets exist inside HDF5
        HDF5.h5open(h5_file, "r") do f
            @test haskey(f, "_meta_")
            mg = f["_meta_"]
            @test haskey(mg, "lesion_annotations.json")
            @test haskey(mg, "organ_mapping")
            @test haskey(mg, "segment_names.json")
            @test haskey(mg, "metadata_presets.json")
            @test haskey(mg, "custom_options.json")
            @test haskey(mg, "custom_fields.json")
            @test haskey(mg, "display_config.json")

            ann = JSON.parse(String(read(mg["lesion_annotations.json"])))
            @test ann["1"]["BaseAnatomy"] == "femur"
            @test ann["1"]["ObservationState"] == "CORRECTED"

            om = JSON.parse(String(read(mg["organ_mapping"])))
            @test om["0"]["1"] == "femur"

            sn = JSON.parse(String(read(mg["segment_names.json"])))
            @test sn["0"]["1"] == "Right Femur"

            pr = JSON.parse(String(read(mg["metadata_presets.json"])))
            @test haskey(pr, "Test HDF5 Preset")
            @test pr["Test HDF5 Preset"]["BaseAnatomy"] == "femur"

            co = JSON.parse(String(read(mg["custom_options.json"])))
            @test "18F-FDG-CustomH5" in co["Radiotracer"]

            cf = JSON.parse(String(read(mg["custom_fields.json"])))
            @test cf["TrialArm"] == "Cohort-A"

            dc = JSON.parse(String(read(mg["display_config.json"])))
            @test dc["label_opacity"] ≈ 0.75
        end

        # Verify loading directly from HDF5 works even if companion JSON is deleted
        rm(json_file; force=true)
        loaded_db = LMW.load_annotations(json_file)
        @test haskey(loaded_db, "1")
        @test loaded_db["1"]["BaseAnatomy"] == "femur"
    end

    # ── 6. Lesion Associations, Matches, and LesionTracker via HDF5 ──────────
    @testset "6. Lesion Associations & LesionTracker Operate Purely on HDF5" begin
        LA.load_matches_from_h5(h5_file)
        @test haskey(LA.get_match_groups(), 1)

        # Link lesion 2 in TP0 to lesion 1 in TP1
        LA.update_match_group!("PET_Lesions_0", 2, "PET_Lesions_1", 1, h5_file)
        LA.save_associations("PET_Lesions_0", "PET_Lesions_1", "Segment_2", ["Segment_1"]; h5_path=h5_file)

        # Verify saved in HDF5
        HDF5.h5open(h5_file, "r") do f
            @test haskey(f["_meta_"], "matches.json")
            @test haskey(f["_meta_"], "lesion_associations.json")
        end

        # Run LesionTracker.track_lesions on tmp_dir (which has NO .nii.gz files, only preprocessed_volumes.h5)
        entries = LT.track_lesions(tmp_dir; h5_path=h5_file)
        @test !isempty(entries)
        tp0_seg1 = filter(e -> e.node_name == "PET_Lesions_0" && e.segment_int == 1, entries)
        @test length(tp0_seg1) == 1
        @test tp0_seg1[1].has_intensity == true
        @test tp0_seg1[1].max_intensity ≈ 12.0

        HDF5.h5open(h5_file, "r") do f
            @test haskey(f["_meta_"], "lesion_tracking_report.json")
        end
    end

    rm(tmp_dir; recursive=true, force=true)
end
ccall(:_exit, Cvoid, (Cint,), 0)
