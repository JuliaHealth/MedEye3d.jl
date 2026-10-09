## Goal Description
The user noticed that when activating Heat-GDT mode and holding left click, a 2D circle appears around the mouse and doesn't grow. This happens because activating Heat-GDT mode also turns on "Edit Mode" (to enable mouse tracking), but we forgot to block the standard 2D CPU paintbrush from executing. The regular paintbrush draws a circle that perfectly overrides/hides the Heat-GDT 3D growth underneath it.

## User Review Required
No major architectural changes. We just need to skip the CPU `react_to_draw` step when Heat-GDT mode is active, so the mouse click only seeds the PDE solver and doesn't draw the 2D brush.

## Proposed Changes

### `src/display/reactingToMouseKeyboard/ReactOnMouseClickAndDrag.jl`
Modify `react_to_draw` to return early if `_heatgdt_mode_active[]` is true, so that we don't apply the manual 2D paintbrush overlay while the GPU PDE solver is working.

#### [MODIFY] src/display/reactingToMouseKeyboard/ReactOnMouseClickAndDrag.jl
```julia
    if !stateObject.valueForMasToSet.is_painting_active || isempty(stateObject.textureToModifyVec)
        return
    end
    
    if _heatgdt_mode_active[]
        return # Skip manual 2D paintbrush when Heat-GDT is active
    end
```

## Verification Plan
1. Manually test that holding left click in Heat-GDT mode no longer shows the flat 2D brush.
2. The user can verify that the circle expands in 3D using the Heat-GDT shader instead of just being the fixed brush.
