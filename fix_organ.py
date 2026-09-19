with open("src/display/LesionMetadataWindow.jl", "r") as f:
    content = f.read()

start_marker = "    # ── Auto-fill BaseAnatomy, Side, and LesionType when organ mapping updates after painting ──"
end_marker = "    @async begin"

match_start = content.find(start_marker)
match_end = content.find(end_marker, match_start)

if match_start != -1 and match_end != -1:
    new_block = """    # ── Auto-fill BaseAnatomy, Side, and LesionType when organ mapping updates after painting ──
    try
        on(_MEH.organ_mapping_updated) do (lid, organ_name)
            lock(ui_lock) do
                push!(ui_queue, () -> begin
                    try
                        @debug "[PAINT→FILL] Received organ_mapping_updated: lid=$lid, organ='$organ_name'"
                        lid == 0 && return  # skip initial value
                        cur_lesion_str = active_lesion_display[]
                        cur_lesion_s = string(cur_lesion_str)
                        m = match(r"^(\d+)", cur_lesion_s)
                        if m === nothing
                            m2 = match(r"New Lesion\\s+(\\d+)", cur_lesion_s)
                            if m2 !== nothing
                                m = m2
                            end
                        end
                        m === nothing && return
                        parsed_lid = tryparse(Int, m.match)
                        if parsed_lid === lid
                            @debug "[PAINT→FILL] Updating UI for current lesion $lid"
                            
                            if haskey(field_widgets, "BaseAnatomy")
                                tb_anatomic = field_widgets["BaseAnatomy"]
                                tb_anatomic.stored_string[] = organ_name
                                notify(tb_anatomic.stored_string)
                                val = parse_lesion_id(active_lesion_id[])
                                if val !== nothing && val > 0
                                    save_state_to_db(val)
                                end
                            end
                            
                            entry = lookup_anatomy(organ_name)
                            if entry !== nothing
                                if haskey(entry, "side") && haskey(field_widgets, "Side")
                                    side = entry["side"]
                                    if side in ["Left", "Right", "Bilateral", "Central"]
                                        field_widgets["Side"].selection[] = side
                                    end
                                end
                                
                                if haskey(entry, "lesion_class")
                                    lc = entry["lesion_class"]
                                    if lc == "Lymph Node Meta"
                                        update_type_buttons("Lymph Node Meta")
                                    elseif lc == "Bone Meta"
                                        update_type_buttons("Bone Meta")
                                    elseif lc == "Solid Organ / Viscera"
                                        update_type_buttons("Local Tumor / Recurrence")
                                    end
                                    if haskey(field_widgets, "Anatomic Location") && field_widgets["Anatomic Location"] isa Menu
                                        if lc in field_widgets["Anatomic Location"].options[]
                                            field_widgets["Anatomic Location"].selection[] = lc
                                        end
                                    end
                                end
                            end
                            notify(active_lesion_id)
                        end
                    catch e
                        @warn "Failed inside organ_mapping_updated UI queue: $e"
                    end
                end)
            end
        end
        @debug "[PAINT→FILL] Successfully registered organ_mapping_updated listener"
    catch e
        @debug "[PAINT→FILL] FAILED to register organ_mapping_updated listener: $e"
        @warn "Failed to register organ_mapping_updated listener: $e"
    end

"""
    
    new_content = content[:match_start] + new_block + content[match_end:]
    with open("src/display/LesionMetadataWindow.jl", "w") as f:
        f.write(new_content)
    print("Patch applied.")
