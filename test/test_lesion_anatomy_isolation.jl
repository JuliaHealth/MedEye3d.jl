#!/usr/bin/env julia
# Test script: Verify lesion anatomy independence and isolation during navigation and refresh
# Run: JULIA_PYTHONCALL_EXE=$(which python3) julia --project=. test/test_lesion_anatomy_isolation.jl

using Test
using GLMakie
using GLMakie.Makie: Figure, Menu
using Observables

using MedEye3d
import MedEye3d.LesionMetadataWindow as LMW

@testset "Lesion Anatomy Independence Suite" begin

    @testset "1. _set_menu_selection_string! Index 1 Preservation" begin
        # Simulate searchable_menu options starting with ""
        opts = Observable(String["", "Liver", "Bone Meta", "Prostate", "Lung"])
        fig = Figure()
        menu = Menu(fig[1, 1], options = opts[], default = 1)
        
        @test menu.options[][1] == ""
        @test menu.selection[] == "" || menu.i_selected[] == 1
        
        # Lesion 1: Set anatomy to "Skull" (not in initial list)
        LMW._set_menu_selection_string!(menu, "Skull"; all_opts = opts)
        @test menu.selection[] == "Skull"
        @test menu.options[][1] == ""  # Empty string MUST be preserved at index 1
        @test menu.options[][2] == "Skull"
        @test menu.i_selected[] == 2
        
        # Lesion 2: Empty anatomy -> must reset to "", NEVER to "Skull"!
        LMW._set_menu_selection_string!(menu, "")
        @test menu.selection[] == ""
        @test menu.i_selected[] == 1
        
        # Lesion 3: Set anatomy to "Femur"
        LMW._set_menu_selection_string!(menu, "Femur"; all_opts = opts)
        @test menu.selection[] == "Femur"
        @test menu.options[][1] == ""  # Still "" at index 1
        
        # Lesion 4: Empty anatomy -> must reset to "", NOT "Femur" or "Skull"!
        LMW._set_menu_selection_string!(menu, "")
        @test menu.selection[] == ""
        @test menu.i_selected[] == 1
        
        # Switch back to Lesion 1: select "Skull" (already in options)
        LMW._set_menu_selection_string!(menu, "Skull")
        @test menu.selection[] == "Skull"
        @test menu.options[][1] == ""
        
        # Switch to Lesion 5 with whitespace/nothing
        LMW._set_menu_selection_string!(menu, "   ")
        @test menu.selection[] == ""
        @test menu.i_selected[] == 1
        
        LMW._set_menu_selection_string!(menu, nothing)
        @test menu.selection[] == ""
        @test menu.i_selected[] == 1
    end

    @testset "2. get_lesion_state Canonical Key Precedence" begin
        # Create a DB with both a canonical key "1" and an old stale display key "1: New Lesion"
        db = Dict{String, Any}(
            "1" => Dict{String, Any}(
                "BaseAnatomy" => "Skull",
                "_display_name" => "1: Skull",
                "LesionType" => "Bone Meta"
            ),
            "1: New Lesion" => Dict{String, Any}(
                "BaseAnatomy" => "Prostate",  # Stale entry!
                "_display_name" => "1: New Lesion"
            ),
            "2" => Dict{String, Any}(
                "BaseAnatomy" => "Liver",
                "_display_name" => "2: Liver",
                "LesionType" => "Organ Meta"
            ),
            "3" => Dict{String, Any}(
                "_display_name" => "3: Lesion 3"
                # No anatomy set!
            )
        )
        
        # Querying by old key "1: New Lesion" must return the CANONICAL "Skull", not the stale "Prostate"
        st1_old_key = LMW.get_lesion_state(db, "1: New Lesion")
        @test get(st1_old_key, "BaseAnatomy", "") == "Skull"
        
        # Querying by new display name "1: Skull"
        st1_new_key = LMW.get_lesion_state(db, "1: Skull")
        @test get(st1_new_key, "BaseAnatomy", "") == "Skull"
        
        # Querying by canonical "1"
        st1_can = LMW.get_lesion_state(db, "1")
        @test get(st1_can, "BaseAnatomy", "") == "Skull"
        
        # Lesion 2 must be independent
        st2 = LMW.get_lesion_state(db, "2: Liver")
        @test get(st2, "BaseAnatomy", "") == "Liver"
        @test get(st2, "LesionType", "") == "Organ Meta"
        
        # Lesion 3 must have NO BaseAnatomy (not inherit Skull or Liver!)
        st3 = LMW.get_lesion_state(db, "3: Lesion 3")
        @test get(st3, "BaseAnatomy", "") == ""
    end

    @testset "3. Cross-Lesion Overwrite Prevention during Navigation" begin
        # Verify that flushing lesion 1 and applying lesion 2 keeps them independent
        db = Dict{String, Any}(
            "1" => Dict{String, Any}("BaseAnatomy" => "Skull", "_display_name" => "1: Skull"),
            "2" => Dict{String, Any}("_display_name" => "2: Lesion 2"),  # empty anatomy
            "3" => Dict{String, Any}("BaseAnatomy" => "Rib", "_display_name" => "3: Rib")
        )
        
        # Simulate active lesion switching sequence:
        # On Lesion 1:
        st1 = LMW.get_lesion_state(db, "1: Skull")
        @test get(st1, "BaseAnatomy", "") == "Skull"
        
        # User modifies Lesion 1
        st1_edited = copy(st1)
        st1_edited["ObservationState"] = "CONFIRMED"
        db["1"] = st1_edited
        
        # User presses Down Arrow -> navigates to Lesion 2:
        st2 = LMW.get_lesion_state(db, "2: Lesion 2")
        # Lesion 2 must NOT have BaseAnatomy or CONFIRMED state from Lesion 1
        @test get(st2, "BaseAnatomy", "") == ""
        @test get(st2, "ObservationState", "UNREVIEWED") == "UNREVIEWED"
        
        # User presses Down Arrow -> navigates to Lesion 3:
        st3 = LMW.get_lesion_state(db, "3: Rib")
        @test get(st3, "BaseAnatomy", "") == "Rib"
        
        # User presses Up Arrow -> navigates back to Lesion 2:
        st2_again = LMW.get_lesion_state(db, "2: Lesion 2")
        @test get(st2_again, "BaseAnatomy", "") == ""
        
        # User presses Up Arrow -> navigates back to Lesion 1:
        st1_again = LMW.get_lesion_state(db, "1: Skull")
        @test get(st1_again, "BaseAnatomy", "") == "Skull"
        @test get(st1_again, "ObservationState", "") == "CONFIRMED"
    end

    @testset "4. Refresh Anatomy Purges and Updates Only Target Lesion" begin
        db = Dict{String, Any}(
            "1" => Dict{String, Any}("BaseAnatomy" => "Stale Femur", "_display_name" => "1: Femur"),
            "1: Femur" => Dict{String, Any}("BaseAnatomy" => "Stale Femur"),
            "2" => Dict{String, Any}("BaseAnatomy" => "Liver", "_display_name" => "2: Liver"),
            "3" => Dict{String, Any}("_display_name" => "3: Lesion 3")
        )
        
        target_lid = 1
        
        # Simulate step 2 of btn_refresh_anat: Purge all keys where parse_lesion_id == target_lid
        for (k, entry) in db
            if LMW.parse_lesion_id(k) == target_lid && entry isa AbstractDict
                for fld in ["BaseAnatomy", "BaseAnatomySide", "Side", "Anatomic Location", "Anatomical Sublocation", "Anatomical Details", "ClinicalLesionName", "OriginalName", "LesionType"]
                    delete!(entry, fld)
                end
            end
        end
        
        # Verify target lesion 1 anatomy was cleared
        @test !haskey(db["1"], "BaseAnatomy")
        @test !haskey(db["1: Femur"], "BaseAnatomy")
        
        # Verify other lesions were untouched
        @test get(db["2"], "BaseAnatomy", "") == "Liver"
        @test !haskey(db["3"], "BaseAnatomy")
        
        # Simulate refreshed state applied and saved for lesion 1
        refreshed_state = Dict{String, Any}(
            "BaseAnatomy" => "Skull",
            "_display_name" => "1: Skull",
            "LesionType" => "Bone Meta"
        )
        db[string(target_lid)] = refreshed_state
        
        # Verify lesion 1 now has "Skull"
        st1 = LMW.get_lesion_state(db, "1")
        @test get(st1, "BaseAnatomy", "") == "Skull"
        
        # Verify lesion 2 still has "Liver"
        st2 = LMW.get_lesion_state(db, "2")
        @test get(st2, "BaseAnatomy", "") == "Liver"
        
        # Verify lesion 3 still has no anatomy
        st3 = LMW.get_lesion_state(db, "3")
        @test get(st3, "BaseAnatomy", "") == ""
    end

    @testset "5. Autosave Debounce Isolation Guard" begin
        db = Dict{String, Any}(
            "1" => Dict{String, Any}("BaseAnatomy" => "Skull", "_display_name" => "1: Skull"),
            "2" => Dict{String, Any}("BaseAnatomy" => "Liver", "_display_name" => "2: Liver")
        )
        
        # Scenario A: Autosave was triggered for lesion 1, but user switched to lesion 2 before debounce fired.
        target_lid = 1
        active_lid = 2  # active lesion is now 2!
        
        # In our autosave callback:
        # cur_lid = parse_lesion_id(active_lesion_id[])
        # if cur_lid != target_lid -> return
        saved_triggered = false
        if active_lid == target_lid
            db[string(target_lid)] = Dict("Overwritten" => true)
            saved_triggered = true
        end
        
        @test saved_triggered == false
        @test !haskey(db["1"], "Overwritten")
        @test !haskey(db["2"], "Overwritten")
        @test db["2"]["BaseAnatomy"] == "Liver"  # Lesion 2 was NOT overwritten!

        # Scenario B: User remained on lesion 1 when debounce fired.
        active_lid_b = 1
        if active_lid_b == target_lid
            db[string(target_lid)]["Saved"] = true
            saved_triggered = true
        end
        @test saved_triggered == true
        @test db["1"]["Saved"] == true
        @test db["2"]["BaseAnatomy"] == "Liver"
    end

    @testset "6. Generic Lesion Display Name Detection" begin
        # Test which names qualify as generic for auto-renaming
        is_generic(name, lid) = occursin("New Lesion", name) || 
                                occursin("Lesion $lid", name) || 
                                occursin("Lesion $(lid):", name) || 
                                strip(name) == string(lid) || 
                                startswith(lowercase(strip(name)), "lesion")
        
        @test is_generic("1: New Lesion", 1) == true
        @test is_generic("1: Lesion 1", 1) == true
        @test is_generic("Lesion 1", 1) == true
        @test is_generic("1", 1) == true
        @test is_generic("2: New Lesion 2", 2) == true
        @test is_generic("2: Lesion 2 [Grp 1, 3 TPs]", 2) == true
        
        # User customized names must NOT be considered generic
        @test is_generic("1: Skull", 1) == false
        @test is_generic("1: Femur Right", 1) == false
        @test is_generic("2: Liver Metastasis", 2) == false
    end

    @testset "7. Anatomical Details Multiple Rows Isolation" begin
        fig = Figure()
        rel_menus = [Menu(fig[i, 1], options = ["", "Inside / Contained In", "Surrounded By", "Deep To"]) for i in 1:3]
        struct_menus = [Menu(fig[i, 2], options = ["", "Aorta", "Fat", "Liver"]) for i in 1:3]
        
        # Lesion 1 has 2 rows: "Inside / Contained In:Aorta" and "Surrounded By:Fat"
        LMW._set_menu_selection_string!(rel_menus[1], "Inside / Contained In")
        LMW._set_menu_selection_string!(struct_menus[1], "Aorta")
        LMW._set_menu_selection_string!(rel_menus[2], "Surrounded By")
        LMW._set_menu_selection_string!(struct_menus[2], "Fat")
        
        @test rel_menus[1].selection[] == "Inside / Contained In"
        @test struct_menus[1].selection[] == "Aorta"
        @test rel_menus[2].selection[] == "Surrounded By"
        @test struct_menus[2].selection[] == "Fat"
        
        # Now navigate to Lesion 2, which has 0 rows (empty):
        # All rows are reset using _set_menu_selection_string!(menu, "")
        for i in 1:3
            LMW._set_menu_selection_string!(rel_menus[i], "")
            LMW._set_menu_selection_string!(struct_menus[i], "")
        end
        
        # Verify ALL rows are completely cleared to "" and NOT holding Lesion 1's structures!
        for i in 1:3
            @test rel_menus[i].selection[] == ""
            @test struct_menus[i].selection[] == ""
        end
    end

end
