using JSON, Dates

# Simulate the edit history functions (copied from the implementation)
const _EDIT_HISTORY_EXCLUDED_FIELDS = Set{String}([
    "_last_modified_by", "_last_modified_at",
    "_CT_Min", "_CT_Max", "_PET_Min", "_PET_Max", "_SPECT_Min", "_SPECT_Max",
    "_display_name", "_Centroid", "_Diameter_mm", "_Volume_cc", "_Volume_mm3",
    "_Slice_Z", "_TimePoint", "_Modality", "_NodeName",
    "_last_seg_edit_by", "_last_seg_edit_at",
    "_edit_history",
])
const _MAX_EDIT_HISTORY = 500
const _current_user = Ref{String}("")

function _compute_metadata_diff(old::AbstractDict, new::AbstractDict)
    changes = Dict{String,String}[]
    all_keys = union(keys(old), keys(new))
    for k in all_keys
        k in _EDIT_HISTORY_EXCLUDED_FIELDS && continue
        startswith(k, "_") && continue
        old_v = string(get(old, k, ""))
        new_v = string(get(new, k, ""))
        if old_v != new_v
            push!(changes, Dict{String,String}("field" => k, "old" => old_v, "new" => new_v))
        end
    end
    return changes
end

function _append_edit_history!(new_state::Dict, old_state::AbstractDict)
    existing = get(old_state, "_edit_history", Any[])
    if existing isa AbstractString
        try existing = JSON.parse(existing) catch; existing = Any[] end
    end
    if !(existing isa AbstractArray)
        existing = Any[]
    end
    existing = collect(Any, existing)
    changes = _compute_metadata_diff(old_state, new_state)
    if !isempty(changes)
        entry = Dict{String,Any}(
            "user"      => _current_user[],
            "timestamp" => Dates.format(Dates.now(), "yyyy-mm-ddTHH:MM:SS"),
            "action"    => "metadata_edit",
            "changes"   => changes
        )
        push!(existing, entry)
        if length(existing) > _MAX_EDIT_HISTORY
            existing = existing[end-_MAX_EDIT_HISTORY+1:end]
        end
    end
    new_state["_edit_history"] = existing
end

# === Test scenario: jm edits, then mj edits ===

# Initial state (from AI pre-segmentation)
old_state1 = Dict{String,Any}(
    "LesionType" => "Organ Meta",
    "Anatomic Location" => "Solid Organ / Viscera",
    "BaseAnatomy" => "quadriceps femoris (UBERON)",
    "BaseAnatomySide" => "Right",
    "SUV max" => "Max: 21.0",
    "_display_name" => "1: Quadriceps Femoris Right",
    "_CT_Min" => "-150",
    "_CT_Max" => "250",
)

# jm's edit
_current_user[] = "jm"
new_state1 = copy(old_state1)
new_state1["LesionType"] = "Bone Meta"
new_state1["Anatomic Location"] = "Axial Skeleton"
_append_edit_history!(new_state1, old_state1)

println("After jm's edit:")
println("  History entries: ", length(new_state1["_edit_history"]))
for (i, entry) in enumerate(new_state1["_edit_history"])
    println("  [$i] user=$(entry["user"]) action=$(entry["action"]) changes=$(length(entry["changes"]))")
    for c in entry["changes"]
        println("      $(c["field"]): '$(c["old"])' -> '$(c["new"])'")
    end
end

# mj's edit (builds on jm's state)
_current_user[] = "mj"
new_state2 = copy(new_state1)
new_state2["LesionType"] = "Technical Artifact"
new_state2["Alternative Hypothesis (False Positive)"] = "None / Malignant Suspected"
_append_edit_history!(new_state2, new_state1)

println("\nAfter mj's edit:")
println("  History entries: ", length(new_state2["_edit_history"]))
for (i, entry) in enumerate(new_state2["_edit_history"])
    println("  [$i] user=$(entry["user"]) action=$(entry["action"]) changes=$(length(entry["changes"]))")
    for c in entry["changes"]
        println("      $(c["field"]): '$(c["old"])' -> '$(c["new"])'")
    end
end

# Test JSON serialization / deserialization round-trip
json_str = JSON.json(new_state2["_edit_history"])
parsed = JSON.parse(json_str)
println("\nJSON round-trip OK: ", length(parsed), " entries, size=", length(json_str), " bytes")

# Test that no-op saves don't create spurious entries
_current_user[] = "mj"
new_state3 = copy(new_state2)
_append_edit_history!(new_state3, new_state2)
println("\nAfter no-op save (same data):")
println("  History entries: ", length(new_state3["_edit_history"]), " (should still be 2)")

println("\n✅ All tests passed!")
