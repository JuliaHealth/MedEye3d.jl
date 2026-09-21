module ConflictChecker

export ValidationIssue, ConflictSeverity, BLOCKING, WARNING, INFORMATIONAL
export run_conflict_checks

@enum ConflictSeverity begin
    BLOCKING       # Cannot sign until resolved
    WARNING        # May sign after acknowledgement
    INFORMATIONAL  # No action required
end

struct ValidationIssue
    severity::ConflictSeverity
    code::String
    message::String
    details::String
end

"""
    run_conflict_checks(report, lesion_data) -> Vector{ValidationIssue}

Compare report content against structured lesion data.
Returns list of issues found.
"""
function run_conflict_checks(report_data::Dict, lesion_entries::Vector{Dict{String,Any}})
    issues = ValidationIssue[]
    
    # Check 1: Bone lesion exists but report might say M0
    has_bone = any(d -> get(d, "BaseAnatomy", "") in ["Bone", "Skeleton", "Os"] || 
                       get(d, "LesionType", "") == "Bone Meta", lesion_entries)
    mitnm = get(report_data, "final_mitnm", "")
    if has_bone && !contains(mitnm, "M1b")
        push!(issues, ValidationIssue(BLOCKING, "BONE_M_MISMATCH",
            "Bone metastasis exists but miTNM does not include M1b",
            "Found bone lesion but miTNM = $mitnm"))
    end
    
    # Check 2: New lesion exists but not flagged
    has_new = any(d -> get(d, "ObservationState", "") == "NEW", lesion_entries)
    if has_new
        push!(issues, ValidationIssue(WARNING, "NEW_LESION_PRESENT",
            "New lesion(s) detected - ensure reported in findings",
            "One or more lesions marked as NEW"))
    end
    
    # Check 3: Unreviewed lesions remain
    unreviewed = filter(d -> get(d, "ObservationState", "UNREVIEWED") == "UNREVIEWED", lesion_entries)
    if !isempty(unreviewed)
        n = length(unreviewed)
        push!(issues, ValidationIssue(BLOCKING, "UNREVIEWED_LESIONS",
            "$n unreviewed lesion(s) remain",
            "Lesions with UNREVIEWED state: $(join([get(d, "LesionName", "?") for d in unreviewed], ", "))"))
    end
    
    # Check 4: Uncertain lesions should be flagged
    uncertain = filter(d -> get(d, "ObservationState", "") == "UNCERTAIN", lesion_entries)
    if !isempty(uncertain)
        push!(issues, ValidationIssue(WARNING, "UNCERTAIN_LESIONS",
            "$(length(uncertain)) uncertain lesion(s) - review before signing",
            "Uncertain lesions should be resolved or explicitly acknowledged"))
    end
    
    # Check 5: Left/Right consistency - check for laterality mismatches
    for d in lesion_entries
        side = get(d, "Side", "")
        name = get(d, "LesionName", "")
        anatomy = get(d, "BaseAnatomy", "")
        if !isempty(side) && side != "Midline" && side != "Bilateral"
            # Check if anatomy contains opposite side reference
            if (side == "Left" && contains(lowercase(anatomy), "right")) ||
               (side == "Right" && contains(lowercase(anatomy), "left"))
                push!(issues, ValidationIssue(WARNING, "LATERALITY_MISMATCH",
                    "Laterality mismatch for $name: Side=$side but anatomy contains opposite",
                    "Check laterality annotation"))
            end
        end
    end
    
    # Check 6: Registration flagged as questionable
    poor_reg = filter(d -> get(d, "RegistrationQC", "GOOD") in ["QUESTIONABLE", "POOR_MANUAL_MATCH", "FAILED_NOT_EVALUABLE"], lesion_entries)
    if !isempty(poor_reg)
        push!(issues, ValidationIssue(INFORMATIONAL, "REGISTRATION_ISSUES",
            "$(length(poor_reg)) lesion(s) with registration quality concerns",
            "Registration flagged for: $(join([get(d, "LesionName", "?") for d in poor_reg], ", "))"))
    end
    
    # Check 7: Missing TMTV (should be > 0 if accepted lesions exist)
    tmtv = get(report_data, "tmtv_cc", 0.0)
    accepted = filter(d -> get(d, "ObservationState", "") in ["ACCEPTED", "CORRECTED", "NEW"], lesion_entries)
    if !isempty(accepted) && tmtv <= 0.0
        push!(issues, ValidationIssue(WARNING, "ZERO_TMTV",
            "TMTV is 0 but accepted lesions exist",
            "$(length(accepted)) accepted/corrected lesions but TMTV = $tmtv cc"))
    end
    
    return issues
end

end # module
