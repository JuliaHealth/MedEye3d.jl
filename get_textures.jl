using MedEye3d

texs = MedEye3d.AppMain.MEH.global_texture_specs
println("Total Panels: ", length(texs))
if length(texs) > 0
    for tex in texs[1]
        println("Texture ", tex.name, ", color: ", tex.color, ", maskContrib: ", tex.maskContribution, ", studyType: ", tex.studyType, ", minMax: ", tex.minAndMaxValue)
    end
end
