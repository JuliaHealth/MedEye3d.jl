using Base.Meta
module GetTex
    import MedEye3d
    function dump()
        open("/tmp/tex_dump.txt", "w") do f
            for (i, pnl) in enumerate(MedEye3d.AppMain.MEH.global_texture_specs)
                for tex in pnl
                    write(f, "Panel $i, Texture $(tex.name), color=$(tex.color), mask=$(tex.colorMask), isMain=$(tex.isMainImage)\n")
                end
            end
        end
    end
end
GetTex.dump()
