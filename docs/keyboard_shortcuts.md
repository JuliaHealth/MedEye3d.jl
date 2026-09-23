# MedEye3d Keyboard Shortcuts & Controls Reference

> **Last Updated**: 2026-09-22 — Full shortcut overhaul: F1-F9 windowing, N/P/S/T/B/Del remapped.

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
| **Left click + drag** | Paint with current brush | — |
| **[ (left bracket)** | Decrease brush width | ✅ Brush slider moves |
| **] (right bracket)** | Increase brush width | ✅ Brush slider moves |

## Mask & Display

| Shortcut | Action | GUI Sync |
|----------|--------|----------|
| **Q (hold)** | Temporarily hide all mask overlays | ✅ Preserves prior manual mask state |
| **P (hold)** | Hide masks (keeps CT and PET/SPECT visible) | ✅ Preserves prior manual mask state |
| **T (hold)** | Show only CT (hides everything else) | ✅ Preserves prior manual mask state |
| **B** | Toggle max anatomy overlay | ✅ Anatomy button toggles |
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

## Compare & Review Mode (M2 Window)

| Shortcut | Action | GUI Sync |
|----------|--------|----------|
| **F** | Toggle flicker mode | ✅ M2 mode dropdown updates |
| **O** | Toggle overlay mode | ✅ M2 mode dropdown updates |
| **V** | Set M2 reference to prior TP | ✅ Right TP dropdown updates |

## Scrolling & Zooming

| Shortcut | Action | GUI Sync |
|----------|--------|----------|
| **Scroll wheel** | Navigate through slices | ✅ |
| **Shift + Scroll** | Zoom in/out | ✅ |
| **Alt + Scroll** | Zoom in/out (alternative) | ✅ |
| **Ctrl + Scroll** | Adjust PET/CT blend ratio | ✅ Blend slider moves |

## Plane Selection

| Shortcut | Action | Notes |
|----------|--------|-------|
| **Space + 1** | Switch to transverse (axial) | ⚠️ Timing-dependent: Space must be released after number |
| **Space + 2** | Switch to coronal | ⚠️ Same timing caveat |
| **Space + 3** | Switch to sagittal | ⚠️ Same timing caveat |

---

## Synchronized Scrolling

Synchronized scrolling is **ON by default**. When enabled, scrolling on any panel scrolls all panels simultaneously. When disabled (press **S** or **C**, or click the "Sync" button), only the panel under the mouse cursor scrolls.

---

## Legacy / Not Working

> These shortcuts exist in the codebase but are non-functional or have been superseded.

| Shortcut | Intended Action | Status |
|----------|----------------|--------|
| **Z** | Undo last action | ❌ Undo history vector never populated |
| **Tab + number** | Set paint value (lesion ID) | ❌ Flag overwrite prevents combo |
| **Tab + Plus/Minus** | Adjust stroke width | ❌ isPlusPressed comparison type bug |
| **Shift + number** | Show mask N | ❌ setTextureVisibility is no-op stub |
| **Ctrl + number** | Hide mask N | ❌ Same no-op stub |
| **Alt + number** | Set mask N active for painting | ⚠️ Internal only, no GUI sync |

---

## Focus Guard

When typing in any Textbox widget (windowing values, notes, comments), the **Space** shortcut is automatically suppressed to prevent accidental lesion advancement.

> **Note**: GLFW keyboard shortcuts (A, R, E, etc.) only fire when the main imaging window has focus. The Makie metadata panel is a separate OS window with its own keyboard handling.
