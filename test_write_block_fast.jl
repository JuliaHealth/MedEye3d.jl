using Dates

# Dummy structs to simulate stateObjects
struct DummyTexture
    name::String
end
struct DummyState
    textureToModifyVec::Vector{DummyTexture}
    onScrollData::Any
end

# Create dummy stateObjects
st = DummyState([DummyTexture("manualModif")], (dataToScroll = [(name="manualModif", dat=Float32[0.0, 1.0, 0.0])],))
stateObjects = [st]

msg = "Test missing points"

try
    open("/tmp/medeye3d_errors.log", "a") do f
        println(f, "$(Dates.now()) [reactToAddAutoPet] ERROR: $msg")
        for (p_idx, st) in enumerate(stateObjects)
            active_paint = !isempty(st.textureToModifyVec) ? st.textureToModifyVec[1].name : "NONE"
            println(f, "  Panel $p_idx active painting texture: $active_paint")
            for dat in st.onScrollData.dataToScroll
                if dat.name == "manualModif" || dat.name == "Mask" || dat.name == "segmentation"
                    nz_count = count(dat.dat .> 0.0f0)
                    if nz_count > 0
                        nz_vals = unique(filter(x -> x > 0.0f0, dat.dat))
                        println(f, "    $(dat.name) contains $nz_count non-zero voxels. Unique values: $nz_vals")
                    else
                        println(f, "    $(dat.name) is empty (0 non-zero voxels)")
                    end
                end
            end
        end
        println(f, "---")
    end
catch e
    println("Failed to write! Error: $e")
    println(sprint(showerror, e, catch_backtrace()))
end
