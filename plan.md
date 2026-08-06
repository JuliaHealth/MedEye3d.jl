## Goal Description
1. Fix the Right Click cross-plane synchronization orientation bug by correctly setting up the volume orientation in the user scripts.
2. Implement Right Click Drag for translating/panning the image.
3. Implement Shift + Scroll for zooming in and out.
4. Document the changes.

## Proposed Changes

### `scripts/run_interactive_mrb.jl` (and other examples)
- **Fix Orientation**: Currently, only the axial volume is reversed (`reverse(ct_vol, dims=2)`), causing a mismatch in coordinate spaces when cross-plane jumping to Coronal or Sagittal views. We will correct the AP orientation for the base volume first, and derive all orthogonal planes from this unified base. This perfectly synchronizes cross-plane jumping without hacking MedEye3d's internal coordinate logic.

### `src/structs/DataStructs.jl`
- **Add Zoom/Pan State**: Add `zoom::Float32 = 1.0`, `panX::Float32 = 0.0`, and `panY::Float32 = 0.0` to `CalcDimsStruct`.

### `src/display/GLFW/DispUtils/StructsManag.jl`
- **Update Texture Rendering**: Modify `getMainVerticies` to adjust the texture coordinates `(s, t)` using the `zoom`, `panX`, and `panY` variables. This natively implements zooming and panning in OpenGL without changing the quad size.
- **Update Mouse Mapping**: Modify `getTextureCoordinatesFromScreen` to inverse-apply the `zoom` and `pan` offsets when mapping screen coordinates to texture coordinates, ensuring accurate voxel targeting even when zoomed or panned.

### `src/display/reactingToMouseKeyboard/ReactOnMouseClickAndDrag.jl`
- **Right-Click Drag for Panning**: Track the previous mouse position during a right-click. If the mouse moves while the right button is held down, update the `panX` and `panY` fields of the active panel, and recompute/upload the vertices to OpenGL to translate the image. The initial right-click press will still trigger a cross-plane jump, but dragging will pan.

### `src/display/reactingToMouseKeyboard/ReactToScroll.jl`
- **Shift + Scroll for Zooming**: Detect if the `Shift` key is held down using `GLFW.GetKey`. If it is, intercept the scroll event to increase or decrease the `zoom` field of the active panel, then recompute and upload the vertices.

### `docs/` or `README.md`
- Document the new controls: Right-Click Drag for Pan, Shift+Scroll for Zoom, Right-Click for Cross-Plane Jump, and Double-Click for Pane Maximization.

## User Review Required
> [!IMPORTANT]
> - By implementing panning on right-click drag, the previous behavior of "scrubbing" slices by dragging the right-click will be replaced. A single right-click will still jump, but holding and dragging will now pan the image.
> - The orientation bug is rooted in how the volumes were prepared in `run_interactive_mrb.jl`. The correct way is to create a unified base volume with the correct Anterior/Posterior orientation, and then use `permutedims` on that base for Coronal and Sagittal views. I will update the example scripts to reflect this best practice.

## Verification Plan
1. Launch `run_interactive_mrb.jl`.
2. Right-click on the posterior part of the body in Transverse; verify Coronal accurately shows the posterior.
3. Hold right-click and move the mouse; verify the image translates (pans).
4. Hold Shift and use the scroll wheel; verify the image zooms in and out.
5. While zoomed and panned, click on a structure and verify that cross-plane jumping still targets the correct anatomical location (verifying `getTextureCoordinatesFromScreen`).
