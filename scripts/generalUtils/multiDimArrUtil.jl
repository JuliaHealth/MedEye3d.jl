using DrWatson
@quickactivate "Probabilistic medical segmentation"
"""
utilities for dealing with multidimensional arrays - it includes calculating the mouse position on matrix
    slicing 3 dimensional arrays etc
"""
module MultiDimArrUtil

using Setfield,  Main.ForDisplayStructs,  Main.DataStructs
export getMainVerticies




```@doc
calculates proper dimensions form main quad display on the basis of data stored in CalcDimsStruct 
some of the values calculated will be needed for futher derivations for example those that will  calculate mouse positions
reurn CalcDimsStruct enriched by new data 
    ```
function getMainVerticies(calcDimStruct::CalcDimsStruct)::CalcDimsStruct
    #corrections that will be added on both sides (in case of height correction top and bottom in case of width correction left and right)
    # to achieve required ratio
    widthCorr=0.0
    heightCorr=0.0

    correCtedWindowQuadHeight= calcDimStruct.avWindHeightForMain
    correCtedWindowQuadWidth = calcDimStruct.avWindWidtForMain
    isWidthToBeCorrected = false
    isHeightToBeCorrected = false

    
    #first one need to check weather current height to width ratio is as we want it  and if not  is it to high or to low
    if(calcDimStruct.avMainImRatio>calcDimStruct.heightToWithRatio ) 
        #if we have to big height to width ratio we need to reduce size of acual quad from top and bottom
        # we know that we would not need to change width  hence we will use the width to calculate the quad height
        correCtedWindowQuadHeight= calcDimStruct.heightToWithRatio * calcDimStruct.avWindWidtForMain
        isHeightToBeCorrected= true
    end# if to heigh
    
    if(calcDimStruct.avMainImRatio<calcDimStruct.heightToWithRatio )
        #if we have to low height to width ratio we need to reduce size of acual quad from left and right
        # we know that we would not need to change height  hence we will use height to calculate the quad height
        correCtedWindowQuadWidth = calcDimStruct.avWindHeightForMain/calcDimStruct.heightToWithRatio 
        isWidthToBeCorrected= true
    end# if to wide

    # now we still need ratio of the resulting quad window size after corrections relative to  total window size
    quadToTotalHeightRatio= correCtedWindowQuadHeight/calcDimStruct.windowHeight
    quadToTotalWidthRatio= correCtedWindowQuadWidth/calcDimStruct.windowWidth

    # original ratios of available space of main image to total window dimensions
    avQuadToTotalHeightRatio= calcDimStruct.avWindHeightForMain/calcDimStruct.windowHeight
    avQuadToTotalWidthRatio= calcDimStruct.avWindWidtForMain/calcDimStruct.windowWidth
    # if those would be equal to corrected ones we would just start from the  bottom left corner and create quad from there - yet now we need to calculate corrections based on the diffrence of quantities just above and corrected ones
    
    if(isHeightToBeCorrected)
        heightCorr=abs(quadToTotalHeightRatio-avQuadToTotalHeightRatio)
    end # if isHeightToBeCorrected

    if(isWidthToBeCorrected)
        widthCorr=abs(quadToTotalWidthRatio-avQuadToTotalWidthRatio)
    end # if isWidthToBeCorrected

    correctedWidthForTextAccounting = (-1+ calcDimStruct.fractionOfMainIm*2)
    #as in OpenGl we start from -1 and end at 1 those ratios needs to be doubled in order to translate them in the OPEN Gl coordinate system yet we will achieve this doubling by just adding the corrections from both sides
    #hence we do not need  to multiply by 2 becose we get from -1 to 1 so total is 2 
      res =  Float32.([
      # positions                  // colors           // texture coords
      correctedWidthForTextAccounting-widthCorr,  1.0-heightCorr, 0.0,   1.0, 0.0, 0.0,   1.0, 1.0,   # top right
      correctedWidthForTextAccounting-widthCorr, -1.0+heightCorr, 0.0,   0.0, 1.0, 0.0,   1.0, 0.0,   # bottom right
      -1.0+widthCorr, -1.0+heightCorr, 0.0,                0.0, 0.0, 1.0,   0.0, 0.0,   # bottom left
      -1.0+widthCorr,  1.0-heightCorr, 0.0,                1.0, 1.0, 0.0,   0.0, 1.0    # top left 
    ])

    windowWidthCorr=Int32(round( (widthCorr/2)*calcDimStruct.windowWidth))
    windowHeightCorr= Int32(round((heightCorr/2)*calcDimStruct.windowHeight))
   
   

    return setproperties(calcDimStruct, (correCtedWindowQuadHeight= correCtedWindowQuadHeight
                                        ,correCtedWindowQuadWidth= correCtedWindowQuadWidth
                                        ,quadToTotalHeightRatio=quadToTotalHeightRatio
                                        ,quadToTotalWidthRatio=quadToTotalWidthRatio
                                        ,widthCorr=widthCorr
                                        ,heightCorr=heightCorr
                                        ,mainImageQuadVert=res
                                        ,mainQuadVertSize = sizeof(res)
                                        ,windowWidthCorr=windowWidthCorr
                                        ,windowHeightCorr=windowHeightCorr
                                        ))
    
    

    end #getMainVerticies


end#MultiDimArrUtil