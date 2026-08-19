# Architecture: OpenGL Threading & Channel-Based Parallelism

MedEye3d renders medical images using raw OpenGL via `ModernGL.jl` and `GLFW.jl`. This document explains **why** a channel-based architecture is necessary and **how** it works.

---

## The OpenGL Single-Thread Constraint

OpenGL contexts are bound to exactly one thread. Every `gl*` call — `glClear`, `glDrawElements`, `glTexSubImage2D`, texture uploads — **must** execute on the same thread that created the context. Calling OpenGL functions from a different thread results in silent corruption or crashes.

Julia's task scheduler can migrate tasks between OS threads at any `yield` point (`sleep`, `take!`, channel operations). This means a naive approach of calling OpenGL from multiple `@async` tasks is fundamentally broken — even if you think you're "on the right thread", Julia may silently move your task to another OS thread.

MedEye3d solves this with two mechanisms:
1. A **single consumer task** that owns the OpenGL context
2. A **global reentrant lock** that serializes OpenGL access between the consumer and the Makie renderloop

---

## The Channel Architecture

### Channel Creation

The core channel is created in [`SegmentationDisplay.jl`](../src/display/GLFW/SegmentationDisplay.jl) line ~788:

```julia
mainMedEye3dInstance = MainMedEye3d(
    channel = Base.Channel{Any}(consumer, 1000; spawn=false),
    ...
)
```

Key parameters:
- **`consumer`**: The function that processes events (defined at line ~663)
- **`1000`**: Buffer capacity — up to 1000 events can be queued before producers block
- **`spawn=false`**: The consumer runs on the **current task's thread**, not a new one. This is critical because it ensures the consumer inherits the interactive thread where the OpenGL context was created.

### The Interactive Thread Requirement

Julia 1.9+ supports an interactive threadpool started with:
```bash
julia --threads 3,1
# or
export JULIA_NUM_THREADS=3,1
```

The `,1` reserves one thread for interactive work. MedEye3d requires this thread because:
- The GLFW window is created on the interactive thread
- The OpenGL context is bound to that thread
- The consumer task must run on the same thread

If the interactive thread is missing, MedEye3d throws an error at module load time (see `src/MedEye3d.jl` lines 92-95).

### Producer → Channel → Consumer Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    PRODUCER SIDE                            │
│  (Any thread — GLFW callbacks, Makie button handlers)       │
│                                                             │
│  GLFW.SetMouseButtonCallback → put!(channel, MouseStruct)   │
│  GLFW.SetScrollCallback      → put!(channel, Int64)         │
│  GLFW.SetKeyCallback         → put!(channel, KeyboardStruct)│
│  on(btn.clicks)              → put!(channel, SomeEvent())   │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ Channel{Any}    │
              │ capacity: 1000  │
              │ spawn: false    │
              └────────┬────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   CONSUMER SIDE                             │
│  (Interactive thread — owns OpenGL context)                 │
│                                                             │
│  while !shouldStop                                          │
│      data = take!(channel)        # blocks until event      │
│      lock(GLOBAL_OPENGL_LOCK) do  # serialize with Makie    │
│          switch_gl_context!(window)                         │
│          on_next!(states, data)   # dispatch by type        │
│          glClear(GL_COLOR_BUFFER_BIT)                       │
│          for state in states                                │
│              activateTextures(...)                          │
│              glDrawElements(GL_TRIANGLES, 6, ...)           │
│          end                                                │
│          GLFW.SwapBuffers(window)                           │
│          glFlush()                                          │
│      end                                                    │
│  end                                                        │
└─────────────────────────────────────────────────────────────┘
```

---

## The `GLOBAL_OPENGL_LOCK`

Defined in [`SegmentationDisplay.jl`](../src/display/GLFW/SegmentationDisplay.jl) line 16:

```julia
const GLOBAL_OPENGL_LOCK = ReentrantLock()
```

Two systems need OpenGL access:
1. **The consumer loop** — renders the medical image panels
2. **The Makie renderloop** — renders the GUI (buttons, labels, text fields)

Both call `gl*` functions but must never do so concurrently. The lock ensures mutual exclusion:

```julia
# In consumer (SegmentationDisplay.jl line ~733):
lock(GLOBAL_OPENGL_LOCK) do
    switch_gl_context!(window)
    on_next!(stateInstances, channelData)
    glClear(...)
    # ... render panels ...
    GLFW.SwapBuffers(window)
end

# In Makie renderloop (SegmentationDisplay.jl line ~47):
lock(GLOBAL_OPENGL_LOCK) do
    GLMakie.GLAbstraction.with_context(screen.glscreen) do
        GLMakie.pollevents(screen, tick_state[])
        GLMakie.renderframe(screen)
        GLFW.SwapBuffers(screen.glscreen)
    end
end
```

Each acquires the lock, switches to its own OpenGL context, renders, swaps buffers, and releases the lock. This interleaving happens ~60 times per second for each system.

---

## The Polling Task

Defined in [`PrepareWindow.jl`](../src/display/GLFW/startModules/PrepareWindow.jl) lines 111-137:

```julia
function createPollingTask(window::GLFW.Window)
    stopChannel = Channel{Bool}(1)
    t = @task begin
        while true
            if isready(stopChannel)
                take!(stopChannel)
                break
            end
            if GLFW.WindowShouldClose(window)
                break
            end
            sleep(0.008)  # ~120 Hz poll rate
        end
    end
    return (t, stopChannel)
end
```

This is a lightweight sentinel that monitors the window-close flag. It does **not** call `GLFW.PollEvents()` — that is handled exclusively by the Makie renderloop's `pollevents()` to ensure correct event ordering (mouseposition must fire before mousebutton).

---

## Mouse Event Coalescing

Fast mouse movement generates many `MouseStruct` events. To prevent input lag, the consumer coalesces stale intermediate moves (lines 708-727):

```julia
if typeof(channelData) == MouseStruct
    # Drain stale moves, but preserve button state transitions
    while !isempty(mainChannel) && typeof(fetch(mainChannel)) == MouseStruct
        peeked = fetch(mainChannel)
        if peeked.isRightButtonDown != channelData.isRightButtonDown ||
           peeked.isLeftButtonDown != channelData.isLeftButtonDown
            break  # button state changed — don't skip this event
        end
        channelData = take!(mainChannel)  # discard stale, keep latest
    end
end
```

For mask painting (left-button drag), all intermediate points are collected into an array so no brush strokes are missed:

```julia
if channelData.isLeftButtonDown && is_painting_active
    mouseStructAggregationArray = [channelData]
    while !isempty(mainChannel) && typeof(fetch(mainChannel)) == MouseStruct
        push!(mouseStructAggregationArray, take!(mainChannel))
    end
    channelData = mouseStructAggregationArray
end
```

---

## Summary

| Component | Thread | Role |
|---|---|---|
| GLFW Callbacks | Any (via GLFW event queue) | `put!()` events into channel |
| Makie Button `on(clicks)` | Any (Makie event loop) | `put!()` events into channel |
| Consumer task | Interactive thread | `take!()` events, call `on_next!()`, render OpenGL |
| Makie renderloop | Spawned task | Renders GUI buttons under `GLOBAL_OPENGL_LOCK` |
| Polling task | `@task` on interactive thread | Monitors window-close flag |
