## Goal Description
The user reported two distinct bugs when entering the "compare volume" mode:
1. The right-hand pane (Panel 5) does not display any image initially.
2. Clicking Coronal or Axial (changing planes) forces images to appear on both sides, but breaks the ability to scroll them independently.

**Analysis of Bug 1 (No Initial Image):**
When `reactToCompareTimePoints` loads the secondary timepoint into Panel 5 and forces a re-render using `reactToScroll(0, ...)`, `reactToScroll` skips updating the Panel 5 textures because its `scrollNumb` is 0 (i.e. `isSliceChanged == false`) and its `currentlyDispDat` was already initialized to a non-null value during startup. Thus, it retains its old hidden state. By manually setting `currentlyDispDat = nothing` for Panels 1 and 5 just before the forced scroll, we can guarantee that `reactToScroll` will bypass the optimization and cleanly re-upload the textures.

**Analysis of Bug 2 (Cannot Scroll after Plane Change):**
Changing planes triggers `ChangePlane.processKeysInfo` which executes `getMainVerticies(...)` to recalculate quad vertices based on standard 4-pane layouts. This completely overwrites the custom `LeftHalf`/`RightHalf` vertices we set for compare mode, and restores the visibility of the inner panels (2, 3, and 4). Because Panel 3 becomes visible again, the system falls back to quadrant-based mouse event routing (TopLeft, TopRight, etc.), which makes Panel 5 (now drawn overlapping the bottom-right) impossible to target with the scroll wheel.
We can fix this by intercepting the end of `reactToChangePlane` to explicitly restore the `LeftHalf`, `RightHalf`, and `Hidden` states if `compare_mode[]` is true.

## Proposed Changes
### src/display/GLFW/MakieEventHandlers.jl
#### [MODIFY] src/display/GLFW/MakieEventHandlers.jl
- Update `reactToChangePlane` to conditionally enforce compare-mode vertices after changing planes.
- Update `reactToCompareTimePoints` to set `currentlyDispDat = nothing` before forcing a re-render.

```diff
@@ -43,6 +43,15 @@
         )
         
         ChangePlane.processKeysInfo(Identity(new_scroll), stateObject, dummy_kb, false)
+        
+        if compare_mode[]
+            if idx == 1
+                updateQuadVertices!(stateObject, :LeftHalf)
+            elseif idx == 5
+                updateQuadVertices!(stateObject, :RightHalf)
+            elseif idx in (2, 3, 4)
+                updateQuadVertices!(stateObject, :Hidden)
+            end
+        end
         
         stateObjects[1].switchIndex = idx
         ReactToScroll.reactToScroll(0, stateObjects, false)
@@ -162,6 +171,8 @@
                 end
                 
                 # Re-render both panels
+                stateObjects[1].currentlyDispDat = nothing
+                stateObjects[5].currentlyDispDat = nothing
                 old_idx = stateObjects[1].switchIndex
                 stateObjects[1].switchIndex = 1
                 ReactToScroll.reactToScroll(0, stateObjects, false)
```

## Verification Plan
1. Apply the multi-replace changes to `MakieEventHandlers.jl`.
2. Ensure the code compiles cleanly (e.g. using a dummy script import).
3. The user will be able to enter compare mode, immediately see both panes, change planes, and still scroll flawlessly.
