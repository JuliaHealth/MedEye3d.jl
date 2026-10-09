# Heat-GDT GPU Interactive Segmentation — Implementation Guide

## Overview

Heat-GDT (Heat Geodesic Distance Transform) is an interactive 3D lesion segmentation tool
that uses GPU-accelerated heat diffusion to grow segmentation masks in real time.  The user
clicks and holds on a lesion, and the mask grows outward from the seed point, guided by a
precomputed diffusivity field derived from image edges.

This document explains every stage of the pipeline in detail: from the moment the user
presses **W** to activate wand mode, through the GPU diffusion loop, to the final commit
of discrete integer labels into the segmentation volume.

---

## Architecture Overview

```
  ┌──────────────────────────────────────────────────────────────────────────┐
  │                        USER INTERACTION                                 │
  │  1. Press W → toggle HeatGDT wand mode ON/OFF                         │
  │  2. Press E → enter edit mode (if not already)                         │
  │  3. Left-click + hold → start growing                                  │
  │  4. Release mouse → commit discrete mask                               │
  └──────────────────────────────────────────────────────────────────────────┘
                          │                    ▲
                          ▼                    │ display_theta (ramped)
  ┌──────────────────────────────────────────────────────────────────────────┐
  │                     EVENT DISPATCH LAYER                                │
  │  HeatGDTStartEvent → reactToHeatGDTStart()                            │
  │  HeatGDTTickEvent  → reactToHeatGDTTick()    (every 50ms)             │
  │  HeatGDTStopEvent  → reactToHeatGDTStop()                             │
  └──────────────────────────────────────────────────────────────────────────┘
                          │                    ▲
                          ▼                    │ render_slice_to_texture!
  ┌──────────────────────────────────────────────────────────────────────────┐
  │                     VULKAN GPU PIPELINE                                 │
  │                                                                         │
  │  Storage Images (R32_SFLOAT, full volume size W×H×D):                  │
  │    img_diffusivity  — precomputed D(x) edge field                      │
  │    img_u_ping       — heat field buffer A (ping-pong)                  │
  │    img_u_pong       — heat field buffer B (ping-pong)                  │
  │    img_mask         — extracted binary mask                            │
  │                                                                         │
  │  Compute Shaders:                                                       │
  │    seed_init.comp       — write 1.0 in sphere around seed              │
  │    diffuse_step.comp    — one step of ∂u/∂t = ∇·(D(x)∇u)             │
  │    extract_mask.comp    — threshold heat → binary mask                 │
  │    slice_heat_to_texture.comp — render 2D slice for display            │
  │                                                                         │
  │  Staging Buffer (16 MB, persistently mapped):                          │
  │    staging_box_buf  — bounding-box readback for CPU commit             │
  └──────────────────────────────────────────────────────────────────────────┘
                          │
                          ▼ apply_box_mask_to_segvol!
  ┌──────────────────────────────────────────────────────────────────────────┐
  │                   CPU SEG_VOL (Julia Array)                             │
  │  Integer labels: seg_vol[x,y,z] = active_lesion_id where heat ≥ θ     │
  │  Shared by all 3 panels (axial, coronal, sagittal)                     │
  │  Synced to TP cache → autosaved to HDF5                                │
  └──────────────────────────────────────────────────────────────────────────┘
```

---

## Detailed Stage-by-Stage Walkthrough

### Stage 1: Mode Activation (W key)

**File:** `MakieEventHandlers.jl`, function `reactToOneShotHeatGDT`

When the user presses **W**:

1. **Toggle ON:**
   - Creates a new lesion entry in the lesion metadata window
   - Enters Edit/Paint mode (calls `reactToEditMode`)
   - Sets `_heatgdt_mode_active[] = true` in `ReactOnMouseClickAndDrag`
   - Syncs the algorithm dropdown to "Heat-GDT"

2. **Toggle OFF:**
   - Sets `_heatgdt_mode_active[] = false`
   - **Refreshes `HEAT_GDT_COMMIT_THETA`** to `heatgdt_theta[]` (default 0.001)
   - **Refreshes `HEAT_GDT_LAST_DISPLAY_THETA`** to `HEAT_DISPLAY_RAMP_START` (0.5)

### Stage 2: Growth Start (Left-click)

**File:** `MakieEventHandlers.jl`, function `reactToHeatGDTStart`

When the user presses the left mouse button while in HeatGDT mode:

1. **Validation:** Checks that GPU is initialized, diffusivity is uploaded, edit mode is active,
   cursor position is valid, and seed is within volume bounds.

2. **State initialization:**
   ```julia
   _vk_heat_seed[] = (cx, cy, cz)          # Store 3D cursor position
   _vk_heat_total_steps[] = 0               # Reset step counter
   _vk_heat_active[] = true                 # Mark grow-session active
   _vk_heat_start_time[] = time()           # Record wall-clock start (for display ramp)
   HEAT_GDT_COMMIT_THETA[] = heatgdt_theta[]  # Refresh commit threshold
   HEAT_GDT_LAST_DISPLAY_THETA[] = HEAT_DISPLAY_RAMP_START  # Reset display theta
   ```

3. **GPU seed + initial diffusion:**
   - `seed_only!(heat_state, vk_ctx, cx, cy, cz; seed_radius=3)` — clears all buffers,
     writes 1.0 in a sphere of radius 3 voxels around the seed
   - `run_incremental_steps!(heat_state, vk_ctx, 6; ...)` — runs 6 initial diffusion
     steps so the lesion is immediately visible

4. **Initial display with theta ramp:**
   - Computes `display_theta = HEAT_DISPLAY_RAMP_START` (0.5 at t=0)
   - Calls `render_slice_to_texture!` with `display_theta` so only the hot center shows
   - This creates a **small initial dot** rather than a large circle

5. **Async tick loop:** Starts a `@async` loop that injects `HeatGDTTickEvent` every 50ms
   into the event channel while `_vk_heat_active[]` remains true.

### Stage 3: Growth Ticks (50ms loop)

**File:** `MakieEventHandlers.jl`, function `reactToHeatGDTTick`

Every 50ms, the tick handler executes:

1. **Guard checks:**
   - `_vk_heat_active[]` must be true
   - `is_left_button_down[]` must be true (immediate stop if mouse released)
   - Debounce: skip if < 50ms since last dispatch

2. **GPU diffusion (50 steps at full speed):**
   ```julia
   VulkanHeatDiffusion.run_incremental_steps!(heat_state, vk_ctx, 50;
       theta=theta_val, copy_box_to_cpu=false)
   ```
   - Uses the **real theta** (`heatgdt_theta[] = 0.001`) for computation
   - Does NOT copy back to CPU (pure GPU loop for speed)
   - Each step runs the `diffuse_step.comp` shader + `extract_mask.comp`

3. **Display theta ramp computation:**
   ```julia
   elapsed = now - _vk_heat_start_time[]
   display_theta = max(theta_val, HEAT_DISPLAY_RAMP_START * exp(-HEAT_DISPLAY_RAMP_RATE * elapsed))
   HEAT_GDT_LAST_DISPLAY_THETA[] = display_theta  # Track for stop-time logging
   ```

   | Time (s) | display_theta | Visual Effect |
   |----------|--------------|---------------|
   | 0.0      | 0.500        | Tiny seed dot |
   | 0.2      | 0.303        | Small growth  |
   | 0.5      | 0.143        | Moderate      |
   | 1.0      | 0.041        | Significant   |
   | 2.0      | 0.003        | Nearly full   |
   | 2.5+     | 0.001        | Full (= real θ) |

4. **GPU slice rendering:**
   ```julia
   VulkanHeatDiffusion.render_slice_to_texture!(heat_state, vk_ctx, vk_tex,
       slice_dim, slice_idx, active_id; theta=display_theta)
   ```
   - Renders with `display_theta` (not real theta) for progressive reveal
   - The `slice_heat_to_texture.comp` shader reads from the 3D heat field and writes
     the active_id label into the 2D texture wherever `heat >= display_theta`

### Stage 4: Growth Stop (Mouse Release)

**File:** `MakieEventHandlers.jl`, function `reactToHeatGDTStop`

When the user releases the mouse button:

1. **Deactivate:** Sets `_vk_heat_active[] = false` (stops the async tick loop)

2. **Snapshot constants:**
   ```julia
   commit_theta = HEAT_GDT_COMMIT_THETA[]        # = 0.001 (the real threshold)
   last_display_theta = HEAT_GDT_LAST_DISPLAY_THETA[]  # what was last shown
   ```

3. **GPU → CPU transfer:**
   ```julia
   VulkanHeatDiffusion.run_incremental_steps!(heat_state, vk_ctx, 0;
       theta=commit_theta, copy_box_to_cpu=true)
   ```
   - Runs 0 additional diffusion steps
   - The `extract_mask` shader runs on the bounding box with `commit_theta`
   - The extracted mask is copied to the persistently-mapped staging buffer via
     `vkCmdCopyImageToBuffer`

4. **Discrete mask commit (float → integer labels):**
   ```julia
   VulkanHeatDiffusion.apply_box_mask_to_segvol!(heat_state, seg_vol, label_val;
       theta=commit_theta)
   ```
   This function reads the staging buffer and writes discrete integer labels:
   ```julia
   for each voxel (x,y,z) in bounding box:
       heat_value = staging_buffer[x,y,z]  # Float32
       if heat_value >= commit_theta:
           seg_vol[x,y,z] = active_lesion_id  # Int16 or Int8
   ```

5. **Panel refresh:** Marks all panels (axial, coronal, sagittal) as dirty so they
   redraw from the updated `seg_vol` with the new discrete labels.

6. **TP cache sync:** Copies `seg_vol` to the timepoint cache for autosave/HDF5
   persistence via `mark_tp_mask_dirty!`.

7. **Refresh constants:**
   ```julia
   HEAT_GDT_COMMIT_THETA[] = heatgdt_theta[]           # Reset to default
   HEAT_GDT_LAST_DISPLAY_THETA[] = HEAT_DISPLAY_RAMP_START  # Reset ramp
   ```

---

## Key Constants and Their Roles

All defined in `MakieEventHandlers.jl`, lines 179–200:

| Constant | Type | Default | Role |
|----------|------|---------|------|
| `heatgdt_theta` | `Ref{Float32}` | 0.001 | Base threshold for heat → mask conversion |
| `HEAT_DISPLAY_RAMP_START` | `Float32` | 0.5 | Initial display theta (high = small dot) |
| `HEAT_DISPLAY_RAMP_RATE` | `Float32` | 2.5 | Exponential decay rate for display ramp |
| `HEAT_GDT_COMMIT_THETA` | `Ref{Float32}` | 0.001 | Threshold used for final GPU→CPU mask commit |
| `HEAT_GDT_LAST_DISPLAY_THETA` | `Ref{Float32}` | 0.5 | Last display theta (tracks what user saw) |
| `HEAT_TICK_MIN_INTERVAL_S` | `Float64` | 0.05 | Debounce interval (50ms = ~20 Hz) |
| `_vk_heat_start_time` | `Ref{Float64}` | 0.0 | Wall-clock start of current interaction |

### Constant Refresh Points

`HEAT_GDT_COMMIT_THETA` and `HEAT_GDT_LAST_DISPLAY_THETA` are refreshed at:

1. **`reactToHeatGDTStart`** — Set to `heatgdt_theta[]` / `HEAT_DISPLAY_RAMP_START`
2. **`reactToHeatGDTStop`** — After commit, reset to defaults
3. **`reactToOneShotHeatGDT` (W toggle OFF)** — Reset to defaults

---

## GPU Memory Layout

The heat diffusion pipeline allocates these GPU resources:

```
img_diffusivity: W×H×D × Float32  (e.g., 512×512×359 = ~373 MB)
img_u_ping:      W×H×D × Float32  (~373 MB)
img_u_pong:      W×H×D × Float32  (~373 MB)
img_mask:        W×H×D × Float32  (~373 MB, for extract_mask output)
staging_box_buf: 16 MB             (persistently mapped, CPU-visible)
────────────────────────────────────────────────
Total GPU:       ~1.5 GB + 16 MB staging
```

The **display theta ramp** adds ZERO additional GPU memory.  It only changes the `theta`
push constant value passed to the existing `slice_heat_to_texture.comp` shader.

---

## The Heat Tensor as Communication Medium

The GPU heat field (`img_u_ping` / `img_u_pong`) serves as the natural communication
tensor between GPU computation and CPU-side discrete labeling:

1. **GPU writes** continuous Float32 values via the `diffuse_step.comp` shader
2. **GPU reads** the heat field for display via `slice_heat_to_texture.comp`
3. **GPU extracts** a mask via `extract_mask.comp` into `img_mask`
4. **GPU copies** the mask bounding box to CPU via `staging_box_buf`
5. **CPU reads** the staging buffer and applies discrete labels to `seg_vol`

The threshold `HEAT_GDT_COMMIT_THETA` controls the boundary between "in the lesion"
and "outside the lesion" during step 5.  All voxels where `heat >= HEAT_GDT_COMMIT_THETA`
become `seg_vol[x,y,z] = active_lesion_id`.

---

## Display Theta Ramp — Fast GPU / Slow Display

The key UX innovation is decoupling GPU computation speed from display speed:

- **GPU computes** diffusion at maximum speed (50 steps per 50ms tick)
- **Display shows** a progressively growing region using a time-ramped theta
- The heat field naturally encodes "arrival time" — center voxels are hottest,
  edge voxels are coolest
- A high theta shows only the hot center; a low theta shows everything

Formula:
```
display_theta = max(real_theta, RAMP_START × exp(-RAMP_RATE × elapsed_seconds))
```

This creates a smooth visual progression from a small dot at click time to the
full segmentation after ~2.5 seconds, regardless of GPU computation speed.

---

## File Map

| File | Role |
|------|------|
| `src/display/GLFW/MakieEventHandlers.jl` | Event handlers, state refs, display ramp logic |
| `src/display/Vulkan/VulkanHeatDiffusion.jl` | GPU pipeline, shader dispatch, staging buffer I/O |
| `src/display/Vulkan/shaders/seed_init.comp` | Initial seed sphere |
| `src/display/Vulkan/shaders/diffuse_step.comp` | Heat diffusion PDE step |
| `src/display/Vulkan/shaders/extract_mask.comp` | Threshold heat → binary mask |
| `src/display/Vulkan/shaders/slice_heat_to_texture.comp` | 2D slice render for display |
| `src/display/reactingToMouseKeyboard/ReactOnMouseClickAndDrag.jl` | Mouse button tracking |
| `src/display/InferenceClient.jl` | Diffusivity precomputation |

---

## Tuning Guide

### Making initial growth smaller
Increase `HEAT_DISPLAY_RAMP_START` (e.g., 0.8).  Higher value means only the very
hottest center voxels pass the display threshold at t=0.

### Making display catch up faster
Increase `HEAT_DISPLAY_RAMP_RATE` (e.g., 4.0).  The display reaches full computation
in `ln(RAMP_START / real_theta) / RAMP_RATE` seconds.

### Changing the commit threshold
Modify `heatgdt_theta[]`.  Lower values include more voxels in the final mask.
The default 0.001 captures essentially everything the diffusion reached.

### Changing growth speed
Modify `steps_per_tick` in `reactToHeatGDTTick` (currently 50).  More steps per tick
= faster total growth but potentially less responsive UI.
