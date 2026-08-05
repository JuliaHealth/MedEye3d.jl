module MakieEvents
export ChangePlaneEvent, CompareTimePointsEvent, ShowSingleLesionEvent
export WindowingEvent, PaintValEvent, SyncLesionEvent
export ChangeTimePointEvent, ToggleLesionEvent, RefreshListEvent
export AddAutoPetEvent, SyncMissingEvent, GenManualEvent
export MapLinkEvent, AutoRunPreprocessEvent, RunPreprocessEvent, ShowBoneMaskEvent, SaveMRBEvent
struct ChangePlaneEvent
    plane :: Symbol # :Axial, :Coronal, :Sagittal
end

struct CompareTimePointsEvent
    compare :: Bool
end

struct ShowSingleLesionEvent
    lesion_id :: Int
end

struct WindowingEvent
    min_val :: Float32
    max_val :: Float32
end

struct PaintValEvent
    val :: Int
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
end
struct SyncMissingEvent end
struct GenManualEvent end

struct MapLinkEvent 
    lesion_id::String
end
struct AutoRunPreprocessEvent
    active :: Bool
end
struct RunPreprocessEvent end
struct ShowBoneMaskEvent
    active :: Bool
end
struct SaveMRBEvent end

end
