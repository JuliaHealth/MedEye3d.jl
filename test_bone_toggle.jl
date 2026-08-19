include("scripts/app/run_interactive_mrb.jl")
using .MedEye3d.SegmentationDisplay.MakieEventHandlers
put!(mainMedEye3dInstance.channel, ShowMaskLayerEvent(2, true))
sleep(5)
for (i, st) in enumerate(mainMedEye3dInstance.stateObjects)
    has_surf = false
    for scr in st.onScrollData.dataToScroll
        if scr.name == "Bone_Surface"
            has_surf = true
            println("Panel $i has Bone_Surface with sum ", sum(scr.dat))
        end
    end
    if !has_surf
        println("Panel $i MISSING Bone_Surface")
    end
end
