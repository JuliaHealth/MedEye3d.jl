path = "src/display/GLFW/MakieEventHandlers.jl"
lines = readlines(path)
out_lines = String[]
in_jump = false
in_line = false
for i in 1:length(lines)
    l = lines[i]
    if occursin("function reactToJumpToMeasurement", l)
        in_jump = true
    elseif occursin("function reactToDeleteMeasurement", l)
        in_jump = false
    elseif occursin("function reactToJumpToLineMeasurement", l)
        in_line = true
    elseif in_line && l == "end" && (i == length(lines) || lines[i+1] == "")
        # end of file or module
        in_line = false
    end

    if (in_jump || in_line) && occursin("for (p_idx, target_slice) in targets", l)
        push!(out_lines, "    ReactToScroll.reactToScrollMultiPanel!(collect(keys(targets)), stateObjects, targets)")
        if in_jump
            push!(out_lines, "    current_viewer_position[] = (cx, cy, cz)")
        else
            push!(out_lines, "    current_viewer_position[] = (mx, my, mz)")
        end
        
        # Skip the rest of the loop block
        global skip_blocks = 1
        continue
    end
    
    if isdefined(Main, :skip_blocks) && skip_blocks > 0
        if occursin("for (p_idx, target_slice) in targets", l)
            skip_blocks += 1
        elseif occursin("end", l)
            # count leading spaces to know when the for loop ends
            if strip(l) == "end" && startswith(l, "    end") && !startswith(l, "        end")
               skip_blocks -= 1
               if skip_blocks == 0
                   continue
               end
            end
        end
        continue
    end
    
    push!(out_lines, l)
end
write(path, join(out_lines, "\n"))
