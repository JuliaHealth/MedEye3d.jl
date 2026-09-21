module ResearchExport

export export_lesion_data_csv, export_lesion_data_json

using Dates

"""
    export_lesion_data_csv(lesion_entries::Vector{Dict{String,Any}}, output_path::String)

Export all lesion annotation data as a CSV file for research analysis.
Columns: case_id, timepoint_index, lesion_id, lesion_name, lesion_type, base_anatomy, 
         side, observation_state, segmentation_origin, has_expert_edits,
         suv_max, suv_mean, volume_cc, diameter_mm, registration_qc,
         registration_comment, reviewer_timestamp, comment
"""
function export_lesion_data_csv(lesion_entries::Vector{Dict{String,Any}}, output_path::String)
    # CSV header
    headers = ["case_id", "timepoint_index", "lesion_id", "lesion_name", 
               "lesion_type", "base_anatomy", "side", "observation_state",
               "segmentation_origin", "has_expert_edits",
               "suv_max", "suv_mean", "volume_cc", "diameter_mm",
               "registration_qc", "registration_comment",
               "reviewer_timestamp", "comment"]
    
    open(output_path, "w") do io
        println(io, join(headers, ","))
        for entry in lesion_entries
            row = [_csv_escape(get(entry, h, "")) for h in headers]
            println(io, join(row, ","))
        end
    end
    return output_path
end

function _csv_escape(val)
    s = string(val)
    if contains(s, ",") || contains(s, "\"") || contains(s, "\n")
        return "\"" * replace(s, "\"" => "\"\"") * "\""
    end
    return s
end

"""
    export_lesion_data_json(lesion_entries::Vector{Dict{String,Any}}, output_path::String)

Export all lesion annotation data as a JSON file.
"""
function export_lesion_data_json(lesion_entries::Vector{Dict{String,Any}}, output_path::String)
    open(output_path, "w") do io
        println(io, "[")
        for (i, entry) in enumerate(lesion_entries)
            # Simple manual JSON serialization
            println(io, "  {")
            keys_list = sort(collect(keys(entry)))
            for (j, k) in enumerate(keys_list)
                v = entry[k]
                comma = j < length(keys_list) ? "," : ""
                if v isa Number
                    println(io, "    \"$k\": $v$comma")
                else
                    println(io, "    \"$k\": \"$(replace(string(v), "\"" => "\\\""))\"$comma")
                end
            end
            comma = i < length(lesion_entries) ? "," : ""
            println(io, "  }$comma")
        end
        println(io, "]")
    end
    return output_path
end

end # module
