module MakieEvents
export ChangePlaneEvent, CompareTimePointsEvent, ShowSingleLesionEvent, ScrollZoomEvent
export WindowingEvent, PaintValEvent, SyncLesionEvent
export ChangeTimePointEvent, ToggleLesionEvent, RefreshListEvent
export AddAutoPetEvent, AIInferenceResultEvent, AIStatusUpdateEvent, SyncMissingEvent, GenManualEvent
export MapLinkEvent, AutoRunPreprocessEvent, RunPreprocessEvent, ShowBoneMaskEvent, ShowMaskLayerEvent, SaveMRBEvent
export CloseWindowEvent, ResizeWindowEvent, SetWindowTitleEvent, ChangeBrushSizeEvent, ToggleMoveLesionModeEvent
export PetBlendEvent, BoneSubsegResultEvent, ScreenshotEvent
struct ChangePlaneEvent
    plane :: Symbol # :Axial, :Coronal, :Sagittal
end

struct CompareTimePointsEvent
    compare :: Bool
end

struct ShowSingleLesionEvent
    lesion_id::Int
end

struct ScrollZoomEvent
    zoom_delta::Float64
end

struct WindowingEvent
    modality :: String
    min_val  :: Float32
    max_val  :: Float32
    WindowingEvent(min_val::Real, max_val::Real) = new("CT", Float32(min_val), Float32(max_val))
    WindowingEvent(modality::String, min_val::Real, max_val::Real) = new(modality, Float32(min_val), Float32(max_val))
end

struct PaintValEvent
    val :: Int
    active :: Bool
    PaintValEvent(val::Int, active::Bool=true) = new(val, active)
end

struct SyncLesionEvent
    lesion_id :: Int
end

struct ChangeTimePointEvent
    change :: Int
end

struct ToggleLesionEvent end
struct RefreshListEvent end

struct AddAutoPetEvent 
    algorithm::String
    channel::Channel{Any}
end
struct AIInferenceResultEvent
    algorithm::String
    active_id::Int
    cx::Int
    cy::Int
    cz::Int
    mask::Union{Nothing, Array{<:Real, 3}}
    seg_vol::Union{Nothing, Array{Float32, 3}}
end
struct AIStatusUpdateEvent
    text::String
end
struct SyncMissingEvent end
struct GenManualEvent
    lesion_id::Int
end

struct MapLinkEvent 
    src_ids::Vector{String}
    dst_ids::Vector{String}
end
struct AutoRunPreprocessEvent
    active :: Bool
end
struct RunPreprocessEvent end
struct ShowBoneMaskEvent
    active :: Bool
end
struct ShowMaskLayerEvent
    layer :: Int
    active :: Bool
end
struct SaveMRBEvent end

struct CloseWindowEvent end
struct ResizeWindowEvent
    width :: Int
    height :: Int
end
struct SetWindowTitleEvent
    title :: String
end

struct ChangeBrushSizeEvent
    size :: Int
end
struct ToggleMoveLesionModeEvent
    active :: Bool
end

struct BoneSubsegResultEvent
    panel_tp::Int
    target_id::Int
    pts_surf::Vector{CartesianIndex{3}}
    pts_marr::Vector{CartesianIndex{3}}
end

struct PetBlendEvent
    weight :: Float32  # 0.0 = CT only, 1.0 = full PET overlay
end

struct ScreenshotEvent
    path::String
    done_channel::Channel{Bool}  # signaled when save completes
end

end
