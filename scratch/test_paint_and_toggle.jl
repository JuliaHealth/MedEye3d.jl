using Test
using MedEye3d
using MedEye3d.ForDisplayStructs
using MedEye3d.DataStructs
using MedEye3d.MakieEvents
import MedEye3d.SegmentationDisplay.MakieEventHandlers as MEH

println("=== Testing ToggleSingleMultiLesionEvent and PaintVal Logic ===")

# Create a mock StateDataFields with a TextureSpec for Mask and a VkPipelineState mock
mutable struct MockVkPipelineState
    ubo_dirty::Bool
end

spec_ct = TextureSpec{Float32}(name="CT", isMainImage=true)
spec_mask = TextureSpec{Int16}(
    name="Mask",
    isMainImage=false,
    isMultiDiscreteMask=true,
    isIntegerTexture=true,
    minAndMaxValue=Int16[1, 1],
    isVisible=true,
    maskContribution=0.6f0
)

# Mock display object
disp_obj = forDisplayObjects(listOfTextSpecifications=[spec_ct, spec_mask], vulkanPipelineState=MockVkPipelineState(false))

mask_3d = zeros(Int16, 64, 64, 10)
scr_mask = ThreeDimRawDat{Int16}(type=Int16, name="Mask", dat=mask_3d)

using Dictionaries

slice_mask_dat = zeros(Int16, 64, 64)
two_dim_mask = TwoDimRawDat{Int16}(Int16, "Mask", slice_mask_dat)
single_slice = SingleSliceDat(
    listOfDataAndImageNames=[two_dim_mask],
    nameIndexes=Dictionary(["Mask"], [1]),
    sliceNumber=5
)

calc_dims = CalcDimsStruct(
    mainQuadVertSize=32,
    mainImageQuadVert=Float32[-1,-1,0,0, 1,-1,0,0, 1,1,0,0, -1,1,0,0, -1,-1,0,0, 1,-1,0,0, 1,1,0,0, -1,1,0,0],
    imageTextureWidth=64,
    imageTextureHeight=64,
    windowWidth=512,
    windowHeight=512
)

scroll_dat = FullScrollableDat(
    dataToScroll=[scr_mask],
    dimensionToScroll=3,
    slicesNumber=10
)

state = StateDataFields(
    mainForDisplayObjects=disp_obj,
    onScrollData=scroll_dat,
    currentlyDispDat=single_slice,
    currentDisplayedSlice=5,
    calcDimsStruct=calc_dims,
    switchIndex=1,
    textureToModifyVec=[spec_mask],
    valueForMasToSet=valueForMasToSetStruct(value=1, is_painting_active=true)
)
states = [state]

# --- TEST 1: ToggleSingleMultiLesionEvent ---
println("\n[TEST 1] Toggle single/multi lesion mode")
MEH.is_single_lesion_mode[] = true
MEH.current_active_lesion_id[] = 2

# Toggle to multi-lesion
MEH.reactToToggleSingleMultiLesion(MakieEvents.ToggleSingleMultiLesionEvent(), states)
@test MEH.is_single_lesion_mode[] == false
@test spec_mask.minAndMaxValue == Int16[1, 10000]
@test disp_obj.vulkanPipelineState.ubo_dirty == true
@test state.isSliceChanged == true
println("  Multi-lesion mode verified: minMax=$(spec_mask.minAndMaxValue), ubo_dirty=$(disp_obj.vulkanPipelineState.ubo_dirty)")

# Reset ubo_dirty
disp_obj.vulkanPipelineState.ubo_dirty = false

# Toggle back to single-lesion
MEH.reactToToggleSingleMultiLesion(MakieEvents.ToggleSingleMultiLesionEvent(), states)
@test MEH.is_single_lesion_mode[] == true
@test spec_mask.minAndMaxValue == Int16[2, 2]
@test disp_obj.vulkanPipelineState.ubo_dirty == true
@test state.isSliceChanged == true
println("  Single-lesion mode verified: minMax=$(spec_mask.minAndMaxValue), ubo_dirty=$(disp_obj.vulkanPipelineState.ubo_dirty)")

# --- TEST 2: New lesion creation and PaintValEvent ---
println("\n[TEST 2] New lesion paint activation")
# Simulate new lesion ID = 3 created
new_id = 3
MEH.reactToShowSingleLesion(MakieEvents.ShowSingleLesionEvent(new_id), states)
@test MEH.is_single_lesion_mode[] == true
@test spec_mask.minAndMaxValue == Int16[3, 3]
@test disp_obj.vulkanPipelineState.ubo_dirty == true
@test state.isSliceChanged == true

disp_obj.vulkanPipelineState.ubo_dirty = false
MEH.reactToPaintVal(MakieEvents.PaintValEvent(new_id, true), states)
@test state.valueForMasToSet.value == 3
@test state.valueForMasToSet.is_painting_active == true
@test spec_mask.isVisible == true
@test spec_mask.minAndMaxValue == Int16[3, 3]
@test disp_obj.vulkanPipelineState.ubo_dirty == true
@test state.isSliceChanged == true
println("  New lesion PaintVal verified: val=$(state.valueForMasToSet.value), minMax=$(spec_mask.minAndMaxValue), ubo_dirty=$(disp_obj.vulkanPipelineState.ubo_dirty)")

# --- TEST 3: react_to_draw with new lesion value ---
println("\n[TEST 3] react_to_draw painting strokes")
mouse = MouseStruct(
    isLeftButtonDown=true,
    lastCoordinates=[CartesianIndex(256, 256), CartesianIndex(260, 260)],
    actualWindowWidth=512,
    actualWindowHeight=512,
    window_id=1
)

disp_obj.vulkanPipelineState.ubo_dirty = false
state.isSliceChanged = false
MedEye3d.ReactOnMouseClickAndDrag.react_to_draw([mouse], states)

painted_voxels = count(==(3), two_dim_mask.dat)
println("  Painted voxels with ID 3: ", painted_voxels)
@test painted_voxels > 0
@test state.isSliceChanged == true

# Shader check
minV = spec_mask.minAndMaxValue[1]
maxV = spec_mask.minAndMaxValue[2]
paint_val = state.valueForMasToSet.value
is_visible_in_shader = (paint_val >= minV - 0.1 && paint_val <= maxV + 0.1)
println("  Shader visibility check for painted value $paint_val with minMax=[$minV, $maxV]: $is_visible_in_shader")
@test is_visible_in_shader == true

println("\n>>> ALL TESTS PASSED SUCCESSFULLY! <<<")
