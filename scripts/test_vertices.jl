using Pkg
Pkg.activate(".")
using MedEye3d
using MedEye3d.ForDisplayStructs
using MedEye3d.DataStructs
using MedEye3d.SegmentationDisplay
using MedEye3d.DispUtils.StructsManag
using MedEye3d.ShadersAndVerticiesForText

fractionOfMainIm = 1.0f0
windowWidth = 1200
windowHeight = 1200
textTexturewidthh = 2000
textTextureheightt = 1
displayMode = QuadImage

println("Testing dimensionToScroll=3 (Axial)")
scrollDims = DataToScrollDims(imageSize=(512, 512, 55), voxelSize=(0.97, 0.97, 5.0), dimensionToScroll=3)
calcDim = CalcDimsStruct(
    windowWidth=windowWidth,
    windowHeight=windowHeight,
    fractionOfMainIm=fractionOfMainIm,
    wordsImageQuadVert=ShadersAndVerticiesForText.getWordsVerticies(fractionOfMainIm),
    wordsQuadVertSize=sizeof(ShadersAndVerticiesForText.getWordsVerticies(fractionOfMainIm)),
    textTexturewidthh=textTexturewidthh,
    textTextureheightt=textTextureheightt
)
calcDim = StructsManag.getHeightToWidthRatio(calcDim, scrollDims)
calcDim1 = StructsManag.getMainVerticies(calcDim, displayMode, 1)
println("Axial (dim=3) Panel 1 widthCorr: ", calcDim1.widthCorr, " heightCorr: ", calcDim1.heightCorr)
println("Axial Panel 1 vertices: ", calcDim1.mainImageQuadVert)

println("\nTesting dimensionToScroll=2 (Coronal)")
scrollDims2 = DataToScrollDims(imageSize=(512, 512, 55), voxelSize=(0.97, 0.97, 5.0), dimensionToScroll=2)
calcDim2 = StructsManag.getHeightToWidthRatio(calcDim, scrollDims2)
calcDim2_panel = StructsManag.getMainVerticies(calcDim2, displayMode, 3)
println("Coronal (dim=2) Panel 3 widthCorr: ", calcDim2_panel.widthCorr, " heightCorr: ", calcDim2_panel.heightCorr)

println("\nTesting dimensionToScroll=1 (Sagittal)")
scrollDims3 = DataToScrollDims(imageSize=(512, 512, 55), voxelSize=(0.97, 0.97, 5.0), dimensionToScroll=1)
calcDim3 = StructsManag.getHeightToWidthRatio(calcDim, scrollDims3)
calcDim3_panel = StructsManag.getMainVerticies(calcDim3, displayMode, 4)
println("Sagittal (dim=1) Panel 4 widthCorr: ", calcDim3_panel.widthCorr, " heightCorr: ", calcDim3_panel.heightCorr)
