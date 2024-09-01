using MedEye3d

# Info: MedEye3d.DataStructs.CalcDimsStruct
# │   imageTextureWidth: Int32 72
# │   imageTextureHeight: Int32 188
# │   textTexturewidthh: Int32 2000
# │   textTextureheightt: Int32 8000
# │   textTextureZeros: Array{UInt8}((2000, 8000)) UInt8[0x00 0x00 … 0x00 0x00; 0x00 0x00 … 0x00 0x00; … ; 0x00 0x00 … 0x00 0x00; 0x00 0x00 … 0x00 0x00]
# │   windowWidth: Int64 1000
# │   windowHeight: Int64 800
# │   fractionOfMainIm: Float32 0.8f0
# │   heightToWithRatio: Float32 2.5f0
# │   avWindWidtForMain: Int32 800
# │   avWindHeightForMain: Int32 800
# │   avMainImRatio: Float32 1.0f0
# │   correCtedWindowQuadHeight: Int32 800
# │   correCtedWindowQuadWidth: Int32 80
# │   quadToTotalHeightRatio: Float32 1.0f0
# │   quadToTotalWidthRatio: Float32 0.08f0
# │   widthCorr: Float32 0.99f0
# │   heightCorr: Float32 0.0f0
# │   windowWidthCorr: Int32 495
# │   windowHeightCorr: Int32 0
# │   mainImageQuadVert: Array{Float32}((32,)) Float32[-0.39, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, -0.39, -1.0  …  0.0, 0.0, -0.010000001, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]
# │   wordsImageQuadVert: Array{Float32}((32,)) Float32[1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, -1.0  …  0.0, 0.0, 0.6, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]
# │   mainQuadVertSize: Int64 128
# └   wordsQuadVertSize: Int64 128

custom_calcdim = MedEye3d.DataStructs.CalcDimsStruct(
    imageTextureWidth=Int32(72),
    imageTextureHeight=Int32(720),
    textTexturewidthh=Int32(2000),
    textTextureheightt=Int32(8000),
    windowWidth=Int64(1000),
    windowHeight=Int64(800),
    fractionOfMainIm=Float32(0.8),
    heightToWithRatio=Float32(0.1),
    avWindWidtForMain=Int32(800),
    avWindHeightForMain=Int32(800),
    avMainImRatio=Float32(1.0),
    correCtedWindowQuadHeight=Int32(800),
    correCtedWindowQuadWidth=Int32(80),
    quadToTotalHeightRatio=Float32(1.0),
    quadToTotalWidthRatio=Float32(0.08),
    widthCorr=Float32(0.99),
    heightCorr=Float32(0.0),
    windowWidthCorr=Int32(495),
    windowHeightCorr=Int32(0),)

MedEye3d.StructsManag.getMainVerticies(custom_calcdim)

# ┌ Info: MedEye3d.DataStructs.CalcDimsStruct
# │   imageTextureWidth: Int32 512
# │   imageTextureHeight: Int32 512
# │   textTexturewidthh: Int32 2000
# │   textTextureheightt: Int32 8000
# │   textTextureZeros: Array{UInt8}((2000, 8000)) UInt8[0x00 0x00 … 0x00 0x00; 0x00 0x00 … 0x00 0x00; … ; 0x00 0x00 … 0x00 0x00; 0x00 0x00 … 0x00 0x00]
# │   windowWidth: Int64 1000
# │   windowHeight: Int64 800
# │   fractionOfMainIm: Float32 0.8f0
# │   heightToWithRatio: Float32 1.0f0
# │   avWindWidtForMain: Int32 800
# │   avWindHeightForMain: Int32 800
# │   avMainImRatio: Float32 1.0f0
# │   correCtedWindowQuadHeight: Int32 800
# │   correCtedWindowQuadWidth: Int32 800
# │   quadToTotalHeightRatio: Float32 1.0f0
# │   quadToTotalWidthRatio: Float32 0.8f0
# │   widthCorr: Float32 0.0f0
# │   heightCorr: Float32 0.0f0
# │   windowWidthCorr: Int32 0
# │   windowHeightCorr: Int32 0
# │   mainImageQuadVert: Array{Float32}((32,)) Float32[0.6, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.6, -1.0  …  0.0, 0.0, -1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]
# │   wordsImageQuadVert: Array{Float32}((32,)) Float32[1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, -1.0  …  0.0, 0.0, 0.6, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]
# │   mainQuadVertSize: Int64 128
# └   wordsQuadVertSize: Int64 128




# Info: MedEye3d.DataStructs.CalcDimsStruct
# │   imageTextureWidth: Int32 72
# │   imageTextureHeight: Int32 188
# │   textTexturewidthh: Int32 2000
# │   textTextureheightt: Int32 8000
# │   textTextureZeros: Array{UInt8}((2000, 8000)) UInt8[0x00 0x00 … 0x00 0x00; 0x00 0x00 … 0x00 0x00; … ; 0x00 0x00 … 0x00 0x00; 0x00 0x00 … 0x00 0x00]
# │   windowWidth: Int64 1000
# │   windowHeight: Int64 800
# │   fractionOfMainIm: Float32 0.8f0
# │   heightToWithRatio: Float32 2.5f0
# │   avWindWidtForMain: Int32 800
# │   avWindHeightForMain: Int32 800
# │   avMainImRatio: Float32 1.0f0
# │   correCtedWindowQuadHeight: Int32 800
# │   correCtedWindowQuadWidth: Int32 80
# │   quadToTotalHeightRatio: Float32 1.0f0
# │   quadToTotalWidthRatio: Float32 0.08f0
# │   widthCorr: Float32 0.99f0
# │   heightCorr: Float32 0.0f0
# │   windowWidthCorr: Int32 495
# │   windowHeightCorr: Int32 0
# │   mainImageQuadVert: Array{Float32}((32,)) Float32[-0.39, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, -0.39, -1.0  …  0.0, 0.0, -0.010000001, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]
# │   wordsImageQuadVert: Array{Float32}((32,)) Float32[1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, -1.0  …  0.0, 0.0, 0.6, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]
# │   mainQuadVertSize: Int64 128
# └   wordsQuadVertSize: Int64 128


function correctRatios(texel_ratio, heightToWidthRatio, windowHeight, corrected_width)

    heightCorr = 0.0
    widthCorr = 0.0

    # Do not modify below this line
    restSpaceHeight = 1 - heightCorr
    restSpaceWidth = 1 - widthCorr
    multipliedHeight = restSpaceHeight * windowHeight #and why are we multiplying with the calcDimStruct.windowHeight specifically?
    mulitipliedWidth = restSpaceWidth * corrected_width #can u explain why me multiply with corrected_width here?
    recalc_texel_ratio = multipliedHeight / mulitipliedWidth

    return restSpaceHeight, restSpaceWidth, multipliedHeight, mulitipliedWidth, recalc_texel_ratio
end


