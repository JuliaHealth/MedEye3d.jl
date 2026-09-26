path = "src/display/LesionMetadataWindow.jl"
lines = readlines(path)
new_lines = String[]
skip = false
for i in 1:length(lines)
    line = lines[i]
    if occursin("# Loading Overlay for Makie", line)
        skip = true
        continue
    end
    if skip && occursin("_lmw_observables[:loading_overlay_txt] = loading_txt", line)
        skip = false
        # We also need to skip the next blank line if it is there
        continue
    end
    if skip
        continue
    end
    
    # Check if we are right after the skip and the line is blank
    if !skip && length(new_lines) > 0 && occursin("_lmw_observables[:loading_overlay_txt]", lines[max(1, i-1)]) && strip(line) == ""
        continue
    end
    
    if line == "    return res" && i+1 <= length(lines) && lines[i+1] == "ult"
        push!(new_lines, "    return result")
        continue
    end
    if line == "ult" && i > 1 && lines[i-1] == "    return res"
        continue
    end
    
    push!(new_lines, line)
end

open(path, "w") do f
    for line in new_lines
        println(f, line)
    end
end
println("Cleaned up by lines!")
