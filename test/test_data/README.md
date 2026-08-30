# MedEye3d Visual Regression Test Data

## Screenshots

Screenshots are captured by `test/test_visual_regression.jl` running against
the test patient data (`data/pat6_pet_only_debug/pat6_pet_only_debug`).

### How to generate
```bash
# In Docker container:
cd /workspaces/MedEye3d.jl
JULIA_NUM_THREADS=auto,1 julia --project=. test/test_visual_regression.jl
```

### Test Scenarios

| # | File | Description |
|---|------|-------------|
| 01 | `01_quad_view_default.png` | Default quad view after startup (axial CT+PET, PET-only, sagittal, coronal) |
| 02 | `02_scroll_mid.png` | Scrolled to middle of volume |
| 03 | `03_zoom_2x.png` | Panel zoomed ~2x via Shift+Scroll |
| 04 | `04_zoom_pan.png` | Zoomed and panned (GPU pan test) |
| 05 | `05_pet_blend_high.png` | PET overlay at 80% blend weight |
| 06 | `06_pet_blend_low.png` | PET overlay at 20% blend weight |
| 07 | `07_ct_window_bone.png` | CT bone window (-500, 1500) |
| 08 | `08_ct_window_soft.png` | CT soft tissue window (40, 400) |
| 09 | `09_ct_window_lung.png` | CT lung window (-1000, -200) |
| 10 | `10_ct_window_default.png` | CT default window (-150, 250) |
| 11 | `11_lesion_filter_1.png` | Showing only lesion ID 1 |
| 12 | `12_lesion_filter_all.png` | Showing all lesions (filter off) |
| 13 | `13_compare_volumes.png` | Compare volumes mode (TP1 vs TP2 side by side) |
| 14 | `14_compare_scroll.png` | Compare mode scrolled 20 slices |
| 15 | `15_compare_off.png` | Compare mode off, back to quad view |
| 16 | `16_bone_mask_on.png` | Bone surface overlay visible |
| 17 | `17_bone_mask_off.png` | Bone surface overlay hidden |
| 18 | `18_sync_lesion_1.png` | View centered on lesion 1 |
| 19 | `19_next_timepoint.png` | After switching to next time point |
| 20 | `20_prev_timepoint.png` | After switching back to previous TP |
| 21 | `21_zoom_reset.png` | After resetting zoom to 1x |
| 22 | `22_pet_window_change.png` | PET window changed (0, 20) |
| 23 | `23_ctrl_scroll_pet.png` | PET blend via Ctrl+Scroll simulation |

### Notes
- All screenshots are captured from the GLFW OpenGL window (not Makie GUI)
- Resolution matches the GLFW window size (typically 1400×700)
- PNG format with RGB color
- Screenshots capture the actual rendered framebuffer via `glReadPixels`
