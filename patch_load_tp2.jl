lines = readlines("src/display/GLFW/MakieEventHandlers.jl")
start_idx = findfirst(l -> occursin("function _load_tp_into_panel!", l), lines)
end_idx = findnext(l -> occursin("newDataToScroll = StructsManag.getThreeDims", l), lines, start_idx)

new_code = """    # Ensure manualModif is inserted at index 2 if it's missing (to match SegmentationDisplay.jl initialization)
    if length(panel_voxels) >= 1 && panel_voxels[1][1] != "manualModif" && (length(panel_voxels) < 2 || panel_voxels[2][1] != "manualModif")
        insert!(panel_voxels, 2, ("manualModif", zeros(Float32, size(panel_voxels[1][2]))))
    end
    
    # Ensure Bone_Surface and Bone_Marrow are present
    has_surf = any(v -> v[1] == "Bone_Surface", panel_voxels)
    has_marr = any(v -> v[1] == "Bone_Marrow", panel_voxels)
    if !has_surf
        push!(panel_voxels, ("Bone_Surface", zeros(Float32, size(panel_voxels[1][2]))))
    end
    if !has_marr
        push!(panel_voxels, ("Bone_Marrow", zeros(Float32, size(panel_voxels[1][2]))))
    end"""

lines[start_idx+7:end_idx-1] .= ""
lines[start_idx+7] = new_code

open("src/display/GLFW/MakieEventHandlers.jl", "w") do f
    for l in lines
        if l != ""
            println(f, l)
        end
    end
end
