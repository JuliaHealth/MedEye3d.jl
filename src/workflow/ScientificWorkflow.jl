module ScientificWorkflow

export LesionObservationState, UNREVIEWED, ACCEPTED, REJECTED, CORRECTED, UNCERTAIN, NEW, RESOLVED
export RegistrationQC
export SegmentationVersion
export LesionObservation
export LesionTrack
export AnnotationWorkflowController
export AuditEvent

export AnnotationWorkflowState, WF_CASE_LOADING, WF_LESION_REVIEW, WF_EDIT_MASK, WF_PROMPT_SEGMENT, WF_REGISTRATION_REVIEW, WF_CASE_QC, WF_CASE_COMPLETE

export ClinicalPhase, PHASE_CASE_SETUP, PHASE_READ, PHASE_ASSESS, PHASE_REPORT_DRAFT, PHASE_VALIDATION, PHASE_SIGNED

export CaseProfile, PROFILE_INITIAL_STAGING, PROFILE_BCR, PROFILE_PRE_RLT, PROFILE_POST_RLT, PROFILE_RESPONSE, PROFILE_GENERAL

@enum CaseProfile begin
    PROFILE_INITIAL_STAGING
    PROFILE_BCR
    PROFILE_PRE_RLT
    PROFILE_POST_RLT
    PROFILE_RESPONSE
    PROFILE_GENERAL
end
@enum ClinicalPhase begin
    PHASE_CASE_SETUP     # Loading, selecting workflow profile
    PHASE_READ           # Image review, segmentation correction, findings
    PHASE_ASSESS         # Staging, response classification
    PHASE_REPORT_DRAFT   # Report generation, editing
    PHASE_VALIDATION     # Conflict checking, QA
    PHASE_SIGNED         # Locked, signed off
end

"""
    AnnotationWorkflowState
"""
@enum AnnotationWorkflowState begin
    WF_CASE_LOADING
    WF_LESION_REVIEW     # Normal viewing/review mode - A/R/U/X shortcuts active
    WF_EDIT_MASK         # Paint/erase mode active - mouse drag = paint/erase
    WF_PROMPT_SEGMENT    # Prompt segmentation active - clicks = prompts
    WF_REGISTRATION_REVIEW  # Registration QC mode
    WF_CASE_QC           # Completion QC
    WF_CASE_COMPLETE     # Case locked
end

"""
    LesionObservationState

Review and lifecycle state of a lesion observation at a specific time point.
"""
@enum LesionObservationState begin
    UNREVIEWED
    ACCEPTED
    REJECTED
    CORRECTED
    UNCERTAIN
    NEW
    RESOLVED
end

"""
    RegistrationQC

Quality control record for co-registration alignment at a specific lesion observation.
"""
struct RegistrationQC
    status::String
    offset_x::Float32
    offset_y::Float32
    offset_z::Float32
    comment::String

    RegistrationQC(status, offset_x, offset_y, offset_z, comment) =
        new(String(status), Float32(offset_x), Float32(offset_y), Float32(offset_z), String(comment))
end

function RegistrationQC(;
    status = "UNREVIEWED",
    offset_x = 0.0f0,
    offset_y = 0.0f0,
    offset_z = 0.0f0,
    comment = ""
)
    RegistrationQC(status, offset_x, offset_y, offset_z, comment)
end

"""
    SegmentationVersion

Represents a single immutable version of a lesion segmentation mask.
"""
struct SegmentationVersion
    version_id::String
    timestamp::String
    author::String
    origin_model::String
    mask_reference::Any

    SegmentationVersion(version_id, timestamp, author, origin_model, mask_reference) =
        new(string(version_id), string(timestamp), string(author), string(origin_model), mask_reference)
end

function SegmentationVersion(;
    version_id = "",
    timestamp = "",
    author = "",
    origin_model = "",
    mask_reference = nothing
)
    SegmentationVersion(version_id, timestamp, author, origin_model, mask_reference)
end

"""
    LesionObservation

State and metrics for a specific lesion track at a specific time point.
"""
mutable struct LesionObservation
    observation_id::String
    timepoint_index::Int
    state::LesionObservationState
    active_mask_version::String
    versions::Vector{SegmentationVersion}
    registration_qc::RegistrationQC
    suv_max::Float32
    suv_mean::Float32
    volume_ml::Float32

    function LesionObservation(
        observation_id,
        timepoint_index,
        state,
        active_mask_version,
        versions,
        registration_qc,
        suv_max,
        suv_mean,
        volume_ml
    )
        vers = [v isa SegmentationVersion ? v : SegmentationVersion(v...) for v in versions]
        new(
            String(observation_id),
            Int(timepoint_index),
            state,
            String(active_mask_version),
            vers,
            registration_qc,
            Float32(suv_max),
            Float32(suv_mean),
            Float32(volume_ml)
        )
    end
end

function LesionObservation(;
    observation_id = "",
    timepoint_index = 0,
    state = UNREVIEWED,
    active_mask_version = "",
    versions = SegmentationVersion[],
    registration_qc = RegistrationQC(),
    suv_max = 0.0f0,
    suv_mean = 0.0f0,
    volume_ml = 0.0f0
)
    LesionObservation(
        observation_id,
        timepoint_index,
        state,
        active_mask_version,
        versions,
        registration_qc,
        suv_max,
        suv_mean,
        volume_ml
    )
end

"""
    LesionTrack

Represents a tracked lesion across longitudinal examination time points.
"""
mutable struct LesionTrack
    track_id::String
    lesion_class::String
    anatomy_label::String
    observations::Dict{Int, LesionObservation}

    LesionTrack(track_id, lesion_class, anatomy_label, observations::Dict{Int, LesionObservation}) =
        new(String(track_id), String(lesion_class), String(anatomy_label), observations)
    LesionTrack(track_id, lesion_class, anatomy_label, observations::Dict) =
        new(String(track_id), String(lesion_class), String(anatomy_label), Dict{Int, LesionObservation}(Int(k) => v for (k, v) in observations))
    LesionTrack(track_id, lesion_class, anatomy_label) =
        new(String(track_id), String(lesion_class), String(anatomy_label), Dict{Int, LesionObservation}())
end

function LesionTrack(;
    track_id = "",
    lesion_class = "",
    anatomy_label = "",
    observations = Dict{Int, LesionObservation}()
)
    LesionTrack(track_id, lesion_class, anatomy_label, observations)
end

"""
    AnnotationWorkflowController

Controller managing the annotation workflow state, track queue, and active selection.
"""
mutable struct AnnotationWorkflowController
    case_id::String
    tracks::Vector{LesionTrack}
    current_track_id::String
    current_tp_idx::Int

    AnnotationWorkflowController(case_id, tracks::AbstractVector{<:LesionTrack}, current_track_id, current_tp_idx) =
        new(String(case_id), collect(LesionTrack, tracks), String(current_track_id), Int(current_tp_idx))
    AnnotationWorkflowController(case_id, tracks::AbstractVector, current_track_id, current_tp_idx) =
        new(String(case_id), collect(LesionTrack, tracks), String(current_track_id), Int(current_tp_idx))
    AnnotationWorkflowController(case_id, tracks::Dict{<:Any, LesionTrack}, current_track_id, current_tp_idx) =
        new(String(case_id), collect(values(tracks)), String(current_track_id), Int(current_tp_idx))
    AnnotationWorkflowController(case_id, tracks, current_track_id, current_tp_idx) =
        new(String(case_id), collect(LesionTrack, tracks), String(current_track_id), Int(current_tp_idx))
end

function AnnotationWorkflowController(;
    case_id = "",
    tracks = LesionTrack[],
    current_track_id = "",
    current_tp_idx = 0
)
    AnnotationWorkflowController(case_id, tracks, current_track_id, current_tp_idx)
end

"""
    AuditEvent

Immutable record of an annotation action for audit trail.
"""
struct AuditEvent
    event_id::String
    timestamp::String  # ISO 8601
    case_id::String
    lesion_track_id::Int
    timepoint_index::Int
    event_type::String  # "ACCEPT", "REJECT", "CORRECT", "UNCERTAIN", "RESOLVED", "NEW_LESION", "EDIT_START", "EDIT_APPLY", "REVERT_TO_AI", "PROMPT_SEGMENT", "NAVIGATE"
    previous_state::String
    new_state::String
    segmentation_version::String
    tool_used::String
    comment::String

    AuditEvent(
        event_id,
        timestamp,
        case_id,
        lesion_track_id,
        timepoint_index,
        event_type,
        previous_state,
        new_state,
        segmentation_version,
        tool_used,
        comment
    ) = new(
        String(event_id),
        String(timestamp),
        String(case_id),
        Int(lesion_track_id),
        Int(timepoint_index),
        String(event_type),
        String(previous_state),
        String(new_state),
        String(segmentation_version),
        String(tool_used),
        String(comment)
    )
end

function AuditEvent(;
    event_id = "",
    timestamp = "",
    case_id = "",
    lesion_track_id = 0,
    timepoint_index = 0,
    event_type = "",
    previous_state = "",
    new_state = "",
    segmentation_version = "",
    tool_used = "",
    comment = ""
)
    AuditEvent(
        event_id,
        timestamp,
        case_id,
        lesion_track_id,
        timepoint_index,
        event_type,
        previous_state,
        new_state,
        segmentation_version,
        tool_used,
        comment
    )
end

end # module ScientificWorkflow
