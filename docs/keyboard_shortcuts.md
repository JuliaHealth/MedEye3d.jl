# MedEye3d Keyboard Shortcuts & Controls Reference

> **Last Updated**: 2026-11-03 — Added **Lesion Deletion** (`Ctrl+Del`, `Ctrl+Backspace`, or `[x]` GUI button) with full **Ctrl+Z undo** support, and **Move Lesion Mode** (`V` key). Fixed crosshair display on bottom row of quad MRI view on M2 second screen.

---

## Lesion Review Workflow

| Shortcut | Action | GUI Sync |
|----------|--------|----------|
| **A** | Accept current lesion → "ACCEPTED" + auto-advance | ✅ State dropdown updates |
| **R** | Reject current lesion → "REJECTED" + auto-advance | ✅ State dropdown updates |
| **U** | Mark uncertain → "UNCERTAIN" | ✅ State dropdown updates |
| **X** | Mark resolved → "RESOLVED" | ✅ State dropdown updates |
| **Shift + R** | Flag registration as "QUESTIONABLE" | ✅ Registration QC dropdown updates |
| **Space** | Next unreviewed lesion *(Makie window only)* | ✅ Lesion dropdown + metadata |

> **Note**: Space only works when the Makie metadata window has focus. It is suppressed when any Textbox has focus (focus guard).

## Lesion Navigation

| Shortcut | Action | GUI Sync |
|----------|--------|----------|
| **Down ↓** | Next lesion in queue | ✅ Lesion dropdown + view centers |
| **Up ↑** | Previous lesion in queue | ✅ Lesion dropdown + view centers |
| **D** | Center view on current lesion (re-sync) | ✅ View re-centers |
| **N** | Create new lesion at cursor position | ✅ Lesion dropdown + enters paint mode |

## Lesion Deletion & Undo

| Shortcut | Action | GUI Sync |
|----------|--------|----------|
| **Ctrl + Del** | Delete the currently active lesion (zeros mask voxels, removes metadata) | ✅ Lesion dropdown + metadata update |
| **Ctrl + Backspace** | Delete the currently active lesion (alternative shortcut) | ✅ Same as Ctrl+Del |
| **[x]** button *(Makie sidebar)* | Delete lesion via GUI — shows "Yes, Delete" / "Cancel" confirmation | ✅ Confirmation row toggles |
| **Ctrl + Z** | Undo last action (restores deleted lesion — mask voxels, metadata, mappings) | ✅ Full restore |

> **Safety**: `Ctrl+Del` and `Ctrl+Backspace` delete immediately without confirmation (from the GLFW viewer window). The `[x]` button in the Makie sidebar shows a confirmation prompt. All deletions can be reversed with **Ctrl+Z** — the undo stack preserves mask voxels, metadata, organ mappings, and segment data.

## Move Lesion Mode

| Shortcut | Action | GUI Sync |
|----------|--------|----------|
| **V** | Toggle move lesion mode — click to reposition the current lesion centroid | ✅ "Move" button turns green |

## Timepoint Navigation

| Shortcut | Action | GUI Sync |
|----------|--------|----------|
| **Alt + Left ←** | Previous timepoint | ✅ TP dropdown + lesion list |
| **Alt + Right →** | Next timepoint | ✅ TP dropdown + lesion list |
| **Home** | Jump to first TP (baseline) | ✅ TP dropdown + lesion list |
| **End** | Jump to last TP | ✅ TP dropdown + lesion list |

## Edit / Paint Mode

| Shortcut | Action | GUI Sync |
|----------|--------|----------|
| **E** | Enter edit/paint mode | ✅ Paint button turns green |
| **Esc** | Cancel edit, return to view mode | ✅ View button turns blue |
| **Del** | Toggle erase mode (brush erases) | ✅ Erase button turns red |
| **G** | Toggle Heat-GDT segmentation mode | ✅ Algorithm dropdown syncs |
| **W** | One-Shot Heat-GDT Wand (Creates new lesion + auto-arms Heat-GDT) | ✅ Auto-labels, enters Edit/Heat-GDT mode |
| **Left click + drag** | Paint with current brush | — |
| **Left click + hold** | Heat-GDT: grow segmentation (when G or W mode active + edit mode) | ✅ Status bar shows K |
| **[ (left bracket)** | Decrease brush width | ✅ Brush slider moves |
| **] (right bracket)** | Increase brush width | ✅ Brush slider moves |

> **Heat-GDT Mode**: Press **G** to toggle Heat-GDT mode, or press **W** for the One-Shot Wand (which creates a new lesion and auto-arms Heat-GDT). When active, hold the left mouse button to run heat diffusion — the longer you hold, the more the segmentation grows (K increases at 30 steps/second). You must be in **edit mode** (press **E** first, unless you used **W**) for the hold to work. If you used **W**, Heat-GDT mode will automatically deactivate when you release the mouse button.

## Lesion Display Modes (Single / Multi)

You can toggle between viewing only the active lesion (Single Lesion Mode) or all lesions simultaneously (Multi Lesion Mode).

| Shortcut | Action | Window Support | GUI Sync |
|----------|--------|----------------|----------|
| **M** | Toggle Single Lesion vs Multi Lesion Mode | Works in **Main Window** and **Makie Window** | ✅ Button text toggles (Single/All) |

> **Note**: In the Makie Metadata Window, the **M** shortcut is suppressed if you are currently typing in any Textbox (focus guard). You can also click the "Single Lesion" / "All Lesions" button in the Makie UI to toggle this mode manually.

## Mask & Display

| Shortcut | Action | GUI Sync |
|----------|--------|----------|
| **K** | Toggle Lesion mask visibility | ✅ Lesion button toggles |
| **J** | Toggle Bone Marrow visibility | ✅ Marrow button toggles |
| **I** | Toggle Bone Surface visibility | ✅ Surf button toggles |
| **Q (hold)** | Temporarily hide all mask overlays | ✅ Preserves prior manual mask state |
| **P (hold)** | Hide masks (keeps CT and PET/SPECT visible) | ✅ Preserves prior manual mask state |
| **T (hold)** | Show only CT (hides everything else) | ✅ Preserves prior manual mask state |
| **B** | Toggle max anatomy overlay | ✅ Anatomy button toggles |
| **H** | Toggle crosshair overlay | ✅ Crosshair button toggles |
| **S** or **C** | Toggle synchronized scrolling | ✅ Sync button updates |

## CT Windowing Presets

| Shortcut | Preset | Range |
|----------|--------|-------|
| **F1** | Soft Tissue | -160 to 240 HU |
| **F2** | Bone | -450 to 1050 HU |
| **F3** | Lung | -1350 to 150 HU |
| **F4** | Brain | -40 to 120 HU |
| **F5** | Liver | -30 to 200 HU |
| **F6** | Mediastinum | -125 to 225 HU |

## PET Windowing Presets

| Shortcut | Preset | Range |
|----------|--------|-------|
| **F7** | SUV 0–5 | 0 to 5 |
| **F8** | SUV 0–10 | 0 to 10 |
| **F9** | SUV 0–15 | 0 to 15 |

## Dedicated MRI Modality Windowing (PI-RADS v2.1 Evidence-Based)

Each MRI modality has its own independent slider in the Makie sidebar under **"MRI Windowing"**:

| Modality | Default Window | Slider Range | Quick Presets | Step Size | Clinical / PI-RADS Standard |
|----------|----------------|--------------|---------------|-----------|-----------------------------|
| **T2** | `[0, 1000]` | `0:10:3000` | `[0-1000]`, `[100-800]`, `[0-1500]` | `±50` | Standard pelvic anatomy, prostate zonal delineation |
| **ADC** | `[0, 2200]` | `0:10:3500` | `[0-2200]`, `[700-2100]`, `[0-3000]` | `±50` | PI-RADS v2.1: `[700-2100]` drives restricted diffusion (<700 $\mu\text{m}^2/\text{s}$) to black |
| **DWI** | `[0, 150]` | `0:5:1500` | `[0-150]`, `[0-100]`, `[0-300]` | `±10` | High b-value ($b \ge 1400$) focal hyperintensity detection |
| **T1** | `[0, 700]` | `0:10:2500` | `[0-700]`, `[100-800]`, `[0-1200]` | `±50` | Hemorrhage exclusion and pelvic lymph node assessment |
| **DCE** | `[0, 500]` | `0:10:2000` | `[0-500]`, `[50-400]`, `[0-1000]` | `±25` | Early wash-in dynamic contrast enhancement |

## MRI Multimodal & Comparison Modes

| Shortcut / Button | Target Display | Action | GUI Sync |
|-------------------|----------------|--------|----------|
| **Y** or **`MRI Multi (Main)`** button | **Main Screen (Panels 1–4)** | Toggle **MRI Multimodal mode on Main Screen**. Displays 4 distinct co-registered MRI modalities (e.g. `T2`, `DWI`, `ADC`, `T1`/`DCE`) in Quad layout. Full interactive annotation tools (Heat-GDT GPU diffusion, 2D paintbrush, erase) are active. Slices, pan/zoom, and plane changes apply synchronously across all 4 panels. Does NOT open Window 2. | ✅ **`MRI Multi (Main)`** button turns blue |
| **Shift + Y** or **Shift + C** or **`MRI Comp (M2)`** button | **M2 Window (Panels 6–9)** | Toggle **MRI Comparison mode in Window 2**. Opens M2 if closed, displaying 4 distinct MRI modalities in Quad layout for cross-study comparison alongside CT/PET on the main window. | ✅ M2 mode dropdown & **`MRI Comp (M2)`** button sync |
| **C** | **M2 Window** | Toggle Compare Curr/Next TP mode (opens M2 window if closed) | ✅ Compare button & M2 mode dropdown update |
| **F** | **M2 Window** | Toggle flicker mode | ✅ M2 mode dropdown updates |
| **O** | **M2 Window** | Toggle interactive lesion mapping mode (all right lesions vs single matched lesion) | ✅ Interactive Map Mode button updates |

### Automatic Cross-Modality Annotation Synchronization
When in MRI Multimodal mode (or viewing multiple MRI sequences), **drawing or editing on any MRI panel immediately and automatically synchronizes the 3D lesion mask to all other MRI modalities** for that patient. 
- Bounding box voxels are copied directly into the in-memory cache (`tp_data_cache`) across all MRI timepoints.
- All target timepoints are flagged as dirty and autosaved to HDF5.
- Vulkan textures on both Window 1 (Main) and Window 2 (M2) update instantly without needing manual re-drawing.

## Scrolling, Zooming & Panning

| Shortcut / Control | Action | GUI Sync |
|----------|--------|----------|
| **Scroll wheel** | Navigate through slices (syncs across all 4 MRI panels in MRI Comparison mode) | ✅ |
| **Shift + Scroll** | Zoom in/out (syncs across all 4 MRI panels in MRI Comparison mode) | ✅ |
| **Alt + Scroll** | Zoom in/out (alternative) | ✅ |
| **Right-click** | Crosshair jump to 3D point across all orthogonal / comparison views | ✅ |
| **Right-click + drag** | Pan image within viewport (syncs across all 4 MRI panels in MRI Comparison mode) | ✅ |
| **Double-click** | Toggle single-panel full-window zoom on/off | ✅ |
| **0** / **Numpad 0** / **Z** / **Shift + F** | **Zoom to Fit** — crops black background padding to fit non-zero anatomy to the viewport, or resets zoom/pan to `1.0×` when already fitted (also exits double-click zoom). Works in both **GLFW Viewer** and **Makie Metadata Window** | ✅ **Fit (0)** button in Viewport |
| **Ctrl + Scroll** | Adjust PET/CT blend ratio | ✅ Blend slider moves |

## Plane Selection

| Shortcut / Button | Action | Notes |
|----------|--------|-------|
| **Axial** / **Coronal** / **Sagittal** *(Buttons)* | Switch viewing plane (Transverse / Coronal / Sagittal) | In **MRI Comparison** mode, switches all 4 MRI panels simultaneously |
| **Space + 1** | Switch to transverse (axial) | ⚠️ Timing-dependent: Space must be released after number |
| **Space + 2** | Switch to coronal | ⚠️ Same timing caveat |
| **Space + 3** | Switch to sagittal | ⚠️ Same timing caveat |

---

## Synchronized Scrolling

Synchronized scrolling is **ON by default**. When enabled, scrolling on any panel scrolls all panels simultaneously. When disabled (press **S** or **C**, or click the "Sync" button), only the panel under the mouse cursor scrolls.

---

## Measurements

### Mode Control

| Shortcut | Action | Notes |
|----------|--------|-------|
| **Shift + M** | Toggle measurement mode ON/OFF | Enter/exit measurement mode |
| **Shift + L** | Cycle sub-mode: Sphere ↔ Line | Switches between SUV sphere and line distance tools |

### SUV Spheres (Sphere sub-mode)

| Shortcut | Action | Notes |
|----------|--------|-------|
| **Shift + Scroll** | Resize measurement sphere radius | 5mm steps when shift held |
| **Left Click** (in sphere mode) | Place sphere measurement | Computes SUV mean/max in the sphere |
| **Hover** over saved sphere | Shows SUV mean/max in cursor info bar | Works in any mode |

### Line Distances (Line sub-mode)

| Shortcut | Action | Notes |
|----------|--------|-------|
| **Left Click** (first, in line mode) | Set start point of line | Green rubberband follows cursor |
| **Left Click** (second, in line mode) | Set end point and save | Computes distance in mm using voxel spacing |

> **GUI**: The "Measurements" panel in the sidebar has:
> - Toggle button (Shift+M) for measurement mode
> - Sub-mode buttons: Sphere (SUV) / Line (cm)
> - Radius slider (sphere mode only)
> - Show/Hide + Clear All buttons
> - Combined measurement list with [>] (jump to) and [x] (delete) per entry

---

## Legacy / Not Working

> These shortcuts exist in the codebase but are non-functional or have been superseded.

| Shortcut | Intended Action | Status |
|----------|----------------|--------|
| **Tab + number** | Set paint value (lesion ID) | ❌ Flag overwrite prevents combo |
| **Tab + Plus/Minus** | Adjust stroke width | ❌ isPlusPressed comparison type bug |
| **Shift + number** | Show mask N | ❌ setTextureVisibility is no-op stub |
| **Ctrl + number** | Hide mask N | ❌ Same no-op stub |
| **Alt + number** | Set mask N active for painting | ⚠️ Internal only, no GUI sync |

---

## Focus Guard

When typing in any Textbox widget (windowing values, notes, comments), the **Space** shortcut is automatically suppressed to prevent accidental lesion advancement.

> **Note**: GLFW keyboard shortcuts (A, R, E, etc.) only fire when the main imaging window has focus. The Makie metadata panel is a separate OS window with its own keyboard handling.
