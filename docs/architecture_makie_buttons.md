# Architecture: Makie Button Wiring

MedEye3d uses a secondary [GLMakie](https://docs.makie.org/stable/) window as its GUI control panel. This document explains how every Makie button is connected to the OpenGL rendering engine via the shared `Channel{Any}`.

---

## Overview

The GUI is created in [`LesionMetadataWindow.jl`](../src/display/LesionMetadataWindow.jl). It receives the same `channel::Base.Channel{Any}` that the OpenGL consumer reads from. Every button follows the same wiring pattern:

```julia
btn = Makie.Button(fig[row, col]; label="Action Name")
on(btn.clicks) do _
    put!(channel, SomeEventStruct(...))
end
```

When the user clicks, Makie fires the `clicks` Observable. The `on()` callback constructs a typed event struct and pushes it into the channel. The OpenGL consumer `take!()`s it and dispatches via `on_next!()`.

---

## Event Types

All event structs are defined in [`MakieEvents.jl`](../src/structs/MakieEvents.jl):

| Event Type | Purpose | Triggered By |
|---|---|---|
| `ChangePlaneEvent(:Axial/:Coronal/:Sagittal)` | Switch viewing plane | Plane buttons |
| `ChangeTimePointEvent(+1/-1)` | Navigate to next/previous timepoint | TP nav buttons |
| `CompareTimePointsEvent()` | Toggle side-by-side TP comparison | Compare button |
| `ScrollZoomEvent(delta)` | Zoom in/out | Shift+Scroll |
| `ToggleLesionEvent()` | Show/hide lesion overlay | Toggle Lesion button |
| `RefreshListEvent()` | Reload lesion list | Refresh button |
| `WindowingEvent(name, min, max)` | Change CT/PET/SPECT window | Windowing presets |
| `ShowBoneMaskEvent(type, visible)` | Toggle bone surface/marrow | Bone visibility buttons |
| `ShowMaskLayerEvent(layer, active)` | Toggle Anatomy/Segmentation visibility | Layer toggle buttons |
| `CloseWindowEvent()` | Graceful shutdown | Window close callback |
| `DoubleClickEvent(x, y, panel)` | Zoom into panel | Double-click on panel |
| `ShowSingleLesionEvent(id)` | Highlight one lesion | Lesion list click |
| `AddAutoPetEvent(algorithm, channel)` | Launch AI segmentation | AI inference buttons |
| `AIInferenceResultEvent(mask, seg_vol, ...)` | Apply AI result to mask + async recompute SUV/volume | AI async callback |
| `SyncLesionEvent(lesion_id)` | Sync display state for lesion | Lesion selection / navigation |
| `GenManualEvent(lesion_id)` | Manual bone subsegmentation + async recompute metrics | Bone subseg button |
| `PaintValEvent(val, is_paint)` | Set paint value (paint/erase mode) | Paint/Erase buttons |

---

## Button Groups

### Navigation Controls

```julia
# Slice navigation
on(btn_ps.clicks) do _; put!(channel, Int64(-1)) end   # Previous slice
on(btn_ns.clicks) do _; put!(channel, Int64(1)) end    # Next slice

# Timepoint navigation
on(btn_pt.clicks) do _; put!(channel, ChangeTimePointEvent(-1)) end  # Previous TP
on(btn_nt.clicks) do _; put!(channel, ChangeTimePointEvent(1)) end   # Next TP

# Compare timepoints (side-by-side overlay)
on(btn_cv.clicks) do _
    put!(channel, CompareTimePointsEvent())
end

# Plane switching
on(btn_ax.clicks) do _; put!(channel, ChangePlaneEvent(:Axial)) end
on(btn_cor.clicks) do _; put!(channel, ChangePlaneEvent(:Coronal)) end
on(btn_sag.clicks) do _; put!(channel, ChangePlaneEvent(:Sagittal)) end
```

### CT Windowing Presets

```julia
on(btn_soft.clicks) do _; apply_ct_win(-160.0, 240.0) end   # Soft tissue
on(btn_bone.clicks) do _; apply_ct_win(-450.0, 1050.0) end  # Bone
on(btn_lung.clicks) do _; apply_ct_win(-1350.0, 150.0) end  # Lung
```

Where `apply_ct_win` constructs and sends a `WindowingEvent`:
```julia
function apply_ct_win(min_val, max_val)
    put!(channel, WindowingEvent("CT", Float32(min_val), Float32(max_val)))
end
```

### PET Windowing Presets

```julia
on(btn_pet_5.clicks)  do _; apply_pet_win(0.0, 5.0) end
on(btn_pet_10.clicks) do _; apply_pet_win(0.0, 10.0) end
on(btn_pet_15.clicks) do _; apply_pet_win(0.0, 15.0) end
```

### SPECT Windowing Presets

```julia
on(btn_spect_5.clicks)  do _; apply_spect_win(0.0, 5.0) end
on(btn_spect_10.clicks) do _; apply_spect_win(0.0, 10.0) end
on(btn_spect_20.clicks) do _; apply_spect_win(0.0, 20.0) end
```

### Lesion Management

```julia
# Navigate between lesions
on(btn_prev.clicks) do _
    # Decrements current lesion index, jumps camera to lesion center
end
on(btn_next.clicks) do _
    # Increments current lesion index, jumps camera to lesion center
end

# Lesion type classification
on(btn_type_prostate.clicks) do _; update_type_buttons("Prostate") end
on(btn_type_bone.clicks)     do _; update_type_buttons("Bone Meta") end
on(btn_type_organ.clicks)    do _; update_type_buttons("Organ Meta") end
on(btn_type_ln.clicks)       do _; update_type_buttons("Lymph Node Meta") end

# AI bone subsegment visibility
on(btn_vis_surface.clicks) do _
    put!(channel, ShowBoneMaskEvent("Bone_Surface", !current_surface_visible))
end
on(btn_vis_marrow.clicks) do _
    put!(channel, ShowBoneMaskEvent("Bone_Marrow", !current_marrow_visible))
end
```

---

## Consumer Dispatch (`on_next!`)

The consumer in [`SegmentationDisplay.jl`](../src/display/GLFW/SegmentationDisplay.jl) calls `on_next!()` which uses Julia's multiple dispatch to route events:

```julia
# In SegmentationDisplay.jl, the consumer loop:
on_next!(stateInstances, channelData)
```

The `on_next!` function dispatches based on the type of `channelData`:

| Type | Handler |
|---|---|
| `Int64` | `ReactToScroll.reactToScroll()` — scroll slices |
| `MouseStruct` | `ReactOnMouseClickAndDrag.reactToMouseDrag()` — windowing, pan, paint |
| `Vector{MouseStruct}` | Same, with aggregated brush strokes |
| `KeyboardStruct` | `ReactToKeyboard.reactToKeyboard()` — keyboard shortcuts |
| `ChangePlaneEvent` | `MakieEventHandlers.reactToChangePlane()` |
| `ChangeTimePointEvent` | `MakieEventHandlers.reactToChangeTimePoint()` |
| `CompareTimePointsEvent` | `MakieEventHandlers.reactToCompareTimePoints()` |
| `ScrollZoomEvent` | `ReactToScroll.reactToScrollZoom()` |
| `WindowingEvent` | `MakieEventHandlers.reactToWindowing()` |
| `ShowBoneMaskEvent` | `MakieEventHandlers.reactToShowBoneMask()` |
| `ToggleLesionEvent` | `MakieEventHandlers.reactToToggleLesion()` |
| `CloseWindowEvent` | Graceful shutdown sequence |

---

## Thread Safety

Makie runs its own renderloop on a spawned task. Button click callbacks fire on whatever thread Makie's event loop runs on. Since `put!()` on a `Base.Channel` is thread-safe, there is no race condition when the callback pushes an event. The consumer processes events sequentially under `GLOBAL_OPENGL_LOCK`, so all OpenGL mutations are serialized.

The key guarantee: **No button handler ever calls OpenGL directly.** They only `put!()` an event struct. The consumer is the sole owner of the OpenGL context.
