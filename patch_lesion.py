with open("src/display/LesionMetadataWindow.jl", "r") as f:
    lines = f.readlines()

new_lines = []
i = 0
while i < len(lines):
    line = lines[i]
    
    if "MetadataWindowResult(fig, ch_ref, ldb=nothing) = new(fig, ch_ref, ldb, Observable(false), Observable(\"Quad View\"))" in line:
        new_lines.append("    set_compare_mode::Observable{Bool}\n")
        new_lines.append("    ui_queue::Vector{Function}\n")
        new_lines.append("    ui_lock::ReentrantLock\n")
        new_lines.append("    MetadataWindowResult(fig, ch_ref, ldb=nothing) = new(fig, ch_ref, ldb, Observable(false), Observable(\"Quad View\"), Observable(false), Function[], ReentrantLock())\n")
        i += 1
        continue
        
    if "obs_m2 = Observable(false)" in line:
        new_lines.append(line)
        new_lines.append("    ui_queue = Function[]\n")
        new_lines.append("    ui_lock = ReentrantLock()\n")
        i += 1
        continue
        
    if "put!(channel, CompareTimePointsEvent(cv_active[]))" in line and "btn_cv.clicks" in "".join(lines[i-6:i]):
        new_lines.append(line)
        new_lines.append("    end\n")
        new_lines.append("    \n")
        new_lines.append("    on(res.set_compare_mode) do val\n")
        new_lines.append("        if cv_active[] != val\n")
        new_lines.append("            cv_active[] = val\n")
        new_lines.append("            btn_cv.buttoncolor[] = cv_active[] ? GRN : BLU_BTN\n")
        new_lines.append("            update_tp_dropdown_visibility!()\n")
        new_lines.append("            sync_tp_menus_to_current!()\n")
        new_lines.append("            put!(channel, CompareTimePointsEvent(cv_active[]))\n")
        new_lines.append("        end\n")
        i += 2 # skip the "end" that belongs to btn_cv
        continue
        
    if "on(_MEH.tp_switched) do _" in line:
        new_lines.append(line)
        new_lines.append("        update_tp_label()\n")
        new_lines.append("        lock(ui_lock) do\n")
        new_lines.append("            push!(ui_queue, () -> begin\n")
        
        # skip 6 lines of cv_active check
        i += 2 # skip update_tp_label
        if "if cv_active[] != _MEH.compare_mode[]" in lines[i]:
            i += 5
            
        # Add the rest up to "end"
        while True:
            if "if load_clinical_info_for_tp! !== nothing" in lines[i]:
                new_lines.append(lines[i])
                new_lines.append(lines[i+1])
                new_lines.append(lines[i+2])
                new_lines.append("            end)\n")
                new_lines.append("        end\n")
                i += 3
                break
            new_lines.append(lines[i])
            i += 1
        continue
        
    if "if cv_active[] && sec_map_lesions[1][]" in line and "map_selected_left" in lines[i+4]:
        new_lines.append(line)
        new_lines.append("            lock(ui_lock) do\n")
        new_lines.append("                push!(ui_queue, () -> begin\n")
        while True:
            if "catch e" in lines[i+1]:
                new_lines.append(lines[i+1])
                new_lines.append(lines[i+2])
                new_lines.append(lines[i+3])
                new_lines.append("                end)\n")
                new_lines.append("            end\n")
                i += 4
                break
            new_lines.append(lines[i+1])
            i += 1
        continue
        
    if "on(_MEH.organ_mapping_updated) do (lid, organ_name)" in line:
        new_lines.append(line)
        new_lines.append("            lock(ui_lock) do\n")
        new_lines.append("                push!(ui_queue, () -> begin\n")
        while True:
            new_lines.append(lines[i+1])
            if "catch e" in lines[i+1]:
                new_lines.append(lines[i+2])
                new_lines.append("                end)\n")
                new_lines.append("            end\n")
                new_lines.append("        end\n")
                i += 3
                break
            i += 1
        continue
        
    if "res = MetadataWindowResult(fig, channel_ref, lesion_db)" in line:
        new_lines.append(line)
        new_lines.append("    res.ui_queue = ui_queue\n")
        new_lines.append("    res.ui_lock = ui_lock\n")
        i += 1
        continue
        
    new_lines.append(line)
    i += 1

with open("src/display/LesionMetadataWindow.jl", "w") as f:
    f.writelines(new_lines)

print("Python patch applied!")
