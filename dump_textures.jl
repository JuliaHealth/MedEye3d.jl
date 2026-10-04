using MedEye3d
using MedEye3d.AppMain

open("/tmp/tex_dump.txt", "w") do f
    if isdefined(MedEye3d.AppMain, :MEH)
        for (i, pnl) in enumerate(MedEye3d.AppMain.MEH.global_texture_specs)
            for tex in pnl
                write(f, "Panel $i, Texture $(tex.name), minMax=$(tex.minAndMaxValue), color=$(tex.color), mask=$(tex.colorMask), isMain=$(tex.isMainImage), isNuc=$(tex.isNuclearMask), isCont=$(tex.isContinuusMask)\n")
            end
        end
    else
        write(f, "MEH not defined in AppMain\n")
    end
end
