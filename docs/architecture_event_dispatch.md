# Architecture: Event Dispatch Lifecycle

This document traces the complete path of a user interaction from physical input to screen update.

---

## Event Lifecycle Overview

```
User Input → GLFW/Makie Callback → Event Struct → put!(channel) → take!(channel)
    → on_next!(states, event) → Handler Function → Texture/State Update
    → glClear → activateTextures → glDrawElements → SwapBuffers → Screen Update
```

Every user interaction follows this exact pipeline. The system never short-circuits — even a simple scroll goes through the channel.

---

## Phase 1: Input Capture (GLFW Callbacks)

GLFW callbacks are registered in the `coordinateDisplay()` function (SegmentationDisplay.jl). Each callback constructs a typed struct and pushes it into the channel.

### Mouse Button Callback
**File**: [`ReactOnMouseClickAndDrag.jl`](../src/display/reactingToMouseKeyboard/ReactOnMouseClickAndDrag.jl)

```julia
function registerMouseClickFunctions(window, calcD, mainChannel)
    GLFW.SetMouseButtonCallback(window, (_, button, action, mods) -> begin
        x, y = GLFW.GetCursorPos(window)
        put!(mainChannel, MouseStruct(
            eventType = :click,
            x = x, y = y,
            isLeftButtonDown  = GLFW.GetMouseButton(window, GLFW.MOUSE_BUTTON_LEFT) == GLFW.PRESS,
            isRightButtonDown = GLFW.GetMouseButton(window, GLFW.MOUSE_BUTTON_RIGHT) == GLFW.PRESS,
            ...
        ))
    end)
end
```

### Scroll Callback
**File**: [`ReactToScroll.jl`](../src/display/reactingToMouseKeyboard/ReactToScroll.jl)

```julia
function registerMouseScrollFunctions(window, mainChannel)
    GLFW.SetScrollCallback(window, (_, xoff, yoff) -> begin
        if GLFW.GetKey(window, GLFW.KEY_LEFT_SHIFT) == GLFW.PRESS
            put!(mainChannel, ScrollZoomEvent(Float64(yoff)))  # Shift+Scroll = Zoom
        else
            put!(mainChannel, Int64(scroll_delta))             # Plain Scroll = Slice nav
        end
    end)
end
```

### Keyboard Callback
**File**: [`ReactToKeyboard.jl`](../src/display/reactingToMouseKeyboard/reactToKeyboard/ReactToKeyboard.jl)

```julia
function registerKeyboardFunctions(window, mainChannel)
    GLFW.SetKeyCallback(window, (_, key, scancode, action, mods) -> begin
        put!(mainChannel, KeyboardStruct(key=key, action=action, ...))
    end)
end
```

---

## Phase 2: Channel Transport

Events sit in the `Channel{Any}(1000)` buffer. The consumer blocks on `take!()` until an event arrives. Multiple events can queue up during a long render cycle — the buffer capacity of 1000 prevents producer blocking under normal usage.

### Mouse Coalescing

Before dispatching, the consumer performs smart coalescing to prevent input lag:

1. **Move coalescing**: If multiple `MouseStruct` events are queued with the same button state, only the latest position is kept. Earlier intermediate positions are discarded.

2. **Button state preservation**: If a queued `MouseStruct` has a different button state (e.g., button released), coalescing stops to preserve the state transition.

3. **Paint aggregation**: During mask painting (left-click drag with painting active), ALL intermediate points are collected into a `Vector{MouseStruct}` so no brush strokes are lost.

---

## Phase 3: Event Dispatch (`on_next!`)

The `on_next!` function uses Julia's multiple dispatch to route each event type to its handler. This is defined in [`SegmentationDisplay.jl`](../src/display/GLFW/SegmentationDisplay.jl):

```julia
# Scroll (Int64)
function on_next!(states, scrollNum::Int64)
    ReactToScroll.reactToScroll(scrollNum, states, false)
end

# Mouse click/drag
function on_next!(states, mouseStruct::MouseStruct)
    ReactOnMouseClickAndDrag.reactToMouseDrag(mouseStruct, states)
end

# Aggregated paint strokes
function on_next!(states, mouseStructs::Vector{MouseStruct})
    for ms in mouseStructs
        ReactOnMouseClickAndDrag.reactToMouseDrag(ms, states)
    end
end

# Keyboard
function on_next!(states, kb::KeyboardStruct)
    ReactToKeyboard.processKeysInfo(kb, states)
end

# Makie events (from GUI buttons)
function on_next!(states, event::ChangePlaneEvent)
    MakieEventHandlers.reactToChangePlane(event, states)
end

function on_next!(states, event::ChangeTimePointEvent)
    MakieEventHandlers.reactToChangeTimePoint(event, states)
end

# ... and so on for each event type
```

---

## Phase 4: Handler Execution

Each handler modifies the rendering state. Common operations include:

### Scroll Handler
Updates the current slice index, extracts the 2D slice from the 3D volume, and uploads new texture data:
```
reactToScroll() → update slice index → extract 2D slice → glTexSubImage2D()
```

### Mouse Drag Handler
- **Left drag**: Adjusts CT/PET windowing levels (min/max) based on cursor delta
- **Right drag**: Pans the viewport by updating `panX`/`panY` in `CalcDimsStruct`
- **Left drag + painting**: Writes voxel values into the mask volume at cursor position

### Timepoint Change Handler
Swaps the entire volume data arrays (CT, PET, Mask) from the cached `all_tps_data` dictionary, then triggers a full re-render of the current slice.

### Windowing Handler
Updates `minAndMaxValue` on the relevant `TextureSpec`, then re-uploads the current slice with the new window applied.

### Async Side Effects (Non-Blocking)

Some handlers spawn async tasks via `Threads.@spawn` that run outside the render cycle. These never block the consumer or GUI:

- **`reactToAIInferenceResult`**: After writing AI mask voxels, calls `invalidate_and_recompute_lesion_metrics_async!()` which spawns an async task to recompute SUV/volume and populate caches.
- **`react_to_draw`**: After mouse release following a paint stroke, spawns the same async metrics recomputation.
- **`reactToGenManual`**: After manual bone subsegmentation, spawns async recomputation.
- **`reactToAddAutoPet`**: The entire AI inference (TCP call + result wait) runs in a spawned thread; only the result event comes back through the channel.

The async recomputation populates `_lesion_suv_cache` and `_volume_cache`, which are then read by `apply_state()` in the metadata panel on the next lesion selection (O(1) cache lookup).

---

## Phase 5: Render Cycle

After every `on_next!()` call, the consumer renders the updated frame:

```julia
lock(GLOBAL_OPENGL_LOCK) do
    switch_gl_context!(window)

    on_next!(stateInstances, channelData)  # Phase 3-4

    if !shouldStop[1]
        glClear(GL_COLOR_BUFFER_BIT)

        for state in stateInstances
            glBindVertexArray(vao[])

            # Render text overlay (slice number, coordinates)
            activateForTextDisp(state.textDispObj...)
            glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, C_NULL)

            # Render main image panel
            reactivateMainObj(state.mainForDisplayObjects...)
            activateTextures(state.mainForDisplayObjects.listOfTextSpecifications)
            glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, C_NULL)
        end

        GLFW.SwapBuffers(window)
        glFlush()
    end
end
```

The `stateInstances` array contains one `StateDataFields` per panel:
- **SingleImage**: 1 state
- **MultiImage**: 2 states (CT + PET side-by-side)
- **QuadImage**: 4 states (Axial + PET-only + Sagittal + Coronal)
- **5-pane**: 5 states (Axial + PET-only + Sagittal + Coronal + Axial-with-mask)

Each state has its own viewport rectangle, texture set, and scroll position.

---

## Complete Example: User Clicks "Next Timepoint" Button

1. **Makie callback fires** (LesionMetadataWindow.jl):
   ```julia
   on(btn_nt.clicks) do _
       put!(channel, ChangeTimePointEvent(1))
   end
   ```

2. **Event enters channel** — `ChangeTimePointEvent(1)` is buffered.

3. **Consumer takes event** (SegmentationDisplay.jl):
   ```julia
   channelData = take!(mainChannel)  # → ChangeTimePointEvent(1)
   ```

4. **Dispatch to handler** via `on_next!`:
   ```julia
   on_next!(states, event::ChangeTimePointEvent) →
       MakieEventHandlers.reactToChangeTimePoint(event, states)
   ```

5. **Handler executes** (MakieEventHandlers.jl):
   - Increments `current_tp_index`
   - Looks up `all_tps_data[new_index]`
   - Swaps all volume arrays in each state's `onScrollData`
   - Calls `reactToScroll(0, states, false)` to re-render current slice with new data

6. **Render cycle**: Consumer clears screen, iterates states, uploads new textures, draws panels, swaps buffers.

7. **Screen updates** — user sees the next timepoint's CT/PET/Mask data.
