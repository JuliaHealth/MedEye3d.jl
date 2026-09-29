using HDF5, JSON

# Create test data with _edit_history
db = Dict{String,Any}(
    "1" => Dict{String,Any}(
        "LesionType" => "Bone Meta",
        "_edit_history" => [
            Dict("user" => "jm", "timestamp" => "2026-09-20T18:49:32", "action" => "metadata_edit",
                 "changes" => [Dict("field" => "LesionType", "old" => "Organ Meta", "new" => "Bone Meta")]),
            Dict("user" => "mj", "timestamp" => "2026-09-28T17:44:29", "action" => "segmentation_edit",
                 "changes" => [Dict("field" => "SegmentationMask", "old" => "", "new" => "EXPERT_CORRECTION")],
                 "tool" => "paint_brush")
        ]
    )
)

# Save to HDF5 (mimicking save_annotations_hdf5)
h5path = "/tmp/test_edit_history.h5"
h5open(h5path, "w") do file
    for (id, lesion_data) in db
        g = create_group(file, string(id))
        for (k, v) in lesion_data
            if v isa AbstractArray
                try
                    write(g, string(k), v)
                catch
                    write(g, string(k), JSON.json(v))
                end
            elseif v isa AbstractDict
                write(g, string(k), JSON.json(v))
            else
                write(g, string(k), string(v))
            end
        end
    end
end
println("Saved to HDF5")

# Load back (mimicking load_annotations_hdf5)
h5open(h5path, "r") do file
    for id in keys(file)
        obj = file[id]
        for k in keys(obj)
            val = read(obj[k])
            if val isa AbstractString && (startswith(strip(val), "{") || startswith(strip(val), "["))
                parsed = JSON.parse(val)
                if k == "_edit_history"
                    println("Loaded _edit_history: $(length(parsed)) entries")
                    for (i, entry) in enumerate(parsed)
                        println("  [$i] user=$(entry["user"]) action=$(entry["action"])")
                    end
                end
            end
        end
    end
end
println("✅ HDF5 round-trip successful!")
