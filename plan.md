## Goal Description
The user reported that scrolling is not working properly in compare volume mode. In our recent sync synchronization logic update inside `ReactToScroll.jl`, we were manually uploading textures to the GPU using `updateTexture` for synced panels. However, this manual upload missed calling `updateImagesDisplayed` which correctly handles updating the on-screen text (like the slice number) and properly orchestrates texture binding. Replacing the manual loop with `updateImagesDisplayed` will ensure that all synchronized panels accurately reflect scrolling changes and slice number text.

## Proposed Changes
### ReactToScroll.jl
Refactor the synchronization blocks in `reactToScroll` to use `updateImagesDisplayed(singleSlDatSync, ...)` instead of the manual `updateTexture` loop.

#### [MODIFY] src/display/reactingToMouseKeyboard/ReactToScroll.jl
```diff
-                            for updateDat in singleSlDatSync.listOfDataAndImageNames
-                                findList = findall((texSpec) -> texSpec.name == updateDat.name, panelState.mainForDisplayObjects.listOfTextSpecifications)
-                                if !isempty(findList)
-                                    texSpec = panelState.mainForDisplayObjects.listOfTextSpecifications[findList[1]]
-                                    transformedDat = applyZoomPan(updateDat.dat, panelState.calcDimsStruct.zoom, panelState.calcDimsStruct.panX, panelState.calcDimsStruct.panY)
-                                    updateTexture(updateDat.type, transformedDat, texSpec, 0, 0, panelState.calcDimsStruct.imageTextureWidth, panelState.calcDimsStruct.imageTextureHeight)
-                                end
-                            end
+                            updateImagesDisplayed(singleSlDatSync, panelState.mainForDisplayObjects, panelState.textDispObj, panelState.calcDimsStruct, panelState.valueForMasToSet, panelState.crosshairFields, panelState.mainRectFields, panelState.displayMode)
```

## Verification Plan
1. The code should parse and execute without `MethodError`.
2. When the user scrolls in compare mode, both panes should visually update and their slice numbers should advance synchronously.
