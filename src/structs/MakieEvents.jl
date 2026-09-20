module MakieEvents
export ChangePlaneEvent, CompareTimePointsEvent, ShowSingleLesionEvent, ScrollZoomEvent, ScrollEvent
export WindowingEvent, PaintValEvent, SyncLesionEvent
export ChangeTimePointEvent, SetTimePointEvent, ToggleLesionEvent, RefreshListEvent
export AddAutoPetEvent, AIInferenceResultEvent, AIStatusUpdateEvent, SyncMissingEvent, GenManualEvent
export MapLinkEvent, AutoRunPreprocessEvent, RunPreprocessEvent, ShowBoneMaskEvent, ShowMaskLayerEvent, SaveMRBEvent
export CloseWindowEvent, ResizeWindowEvent, SetWindowTitleEvent, ChangeBrushSizeEvent, ToggleMoveLesionModeEvent
export PetBlendEvent, BoneSubsegResultEvent, ScreenshotEvent, LabelOpacityEvent, SyncViewsEvent, LaunchM2Event
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
    window_id::Int
    ScrollZoomEvent(z::Float64, w::Int=1) = new(z, w)
end

struct ScrollEvent
    scroll_delta::Int
    window_id::Int
    ScrollEvent(d::Int, w::Int=1) = new(d, w)
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

struct SetTimePointEvent
    tp_index::Int
    panel::Int  # 0 = single/all, 1 = left panel (compare), 5 = right panel (compare)
    SetTimePointEvent(tp_index::Int, panel::Int=0) = new(tp_index, panel)
end

struct ToggleLesionEvent end
struct RefreshListEvent end

struct AddAutoPetEvent 
    algorithm::String
    channel::Any  # Channel{Any} or ChannelProxy (parallel startup)
end
struct AIInferenceResultEvent
    algorithm::String
    active_id::Int
    cx::Int
    cy::Int
    cz::Int
    mask::Union{Nothing, Array{<:Real, 3}}
    seg_vol::Any
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
    width :: Int      # GLFW window size (for coordinate mapping — cursor coords are in this space)
    height :: Int
    fb_width :: Int   # Framebuffer size (for Vulkan swapchain — actual pixel dimensions)
    fb_height :: Int
    window_id :: Int  # 1 for Main Window, 2 for M2 Window
end
# Backward-compatible constructors
ResizeWindowEvent(w::Int, h::Int) = ResizeWindowEvent(w, h, w, h, 1)
ResizeWindowEvent(w::Int, h::Int, fb_w::Int, fb_h::Int) = ResizeWindowEvent(w, h, fb_w, fb_h, 1)
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
    window_id :: Int
    PetBlendEvent(w::Float32, win::Int=0) = new(w, win)
end

struct ScreenshotEvent
    path::String
    done_channel::Channel{Bool}  # signaled when save completes
end

struct LabelOpacityEvent
    opacity :: Float32  # 0.0 = completely transparent, 1.0 = fully opaque
end


struct SyncViewsEvent
    is_synced::Bool
end

struct LaunchM2Event
    tp_index::Int
    window::Any
    mode::String
    LaunchM2Event(tp_index::Int, window::Any=nothing, mode::String="Pure PET (Current TP)") = new(tp_index, window, mode)
end

end # module MakieEvents
