using DrWatson
@quickactivate "Julia Med 3d"

using ColorTypes

"""
utilities for dealing data structs like FullScrollableDat or SingleSliceDat
"""
module StructsManag
using  Setfield,   ..ForDisplayStructs,   ..DataStructs, Rocket
export getThreeDims,addToforUndoVector,cartTwoToThree,getHeightToWidthRatio,threeToTwoDimm,modSlice!, threeToTwoDimm,modifySliceFull!,getSlicesNumber,getMainVerticies

```@doc
given two dim dat it sets points in given coordinates in given slice to given value
coords - coordinates in a plane of chosen slice to modify
value - value to set for given points
return reference to modified slice
```
function modSlice!(data::TwoDimRawDat{T}
                ,coords::Vector{CartesianIndex{2}}
                ,value::T ) where {T}
   data.dat[coords].=value
   data
end#modSlice



```@doc
gives access to the slice of intrest - way of slicing is defined at the begining
typ - type of data 
slice - slice we want to access
sliceDim - on the basis of what dimension we are slicing
return 2 dimensional array  wrapper -TwoDimRawDat  object representing slice of given 3 dimensional array
!! important returned TwoDimRawDat holds view to the original 3 dimensional data  
```
function threeToTwoDimm(typ::Type{T}
                ,sliceInner::Int
                ,sliceDim::Int
                ,threedimDat::ThreeDimRawDat{T})::TwoDimRawDat{T} where {T}
         
maxSlice =size(threedimDat.dat)[sliceDim]
slice= sliceInner 
if(sliceInner>maxSlice) slice=maxSlice   end
               return TwoDimRawDat{T}(typ,threedimDat.name,selectdim(threedimDat.dat, sliceDim, slice)   )
end#ThreeToTwoDimm


modifySliceFull!Str= """
modifies given slice in given coordinates of given data - queried by name
data - full data we work on and modify
coords - coordinates in a plane of chosen slice to modify (so list of x and y coords)
value - value to set for given points
return reference to modified slice
"""
@doc modifySliceFull!Str
function modifySliceFull!(data::FullScrollableDat
                        ,slice::Int
                        ,coords::Vector{CartesianIndex{2}}
                        ,name::String
                        ,value)
                      
     threeDimDat=data.nameIndexes[name] |>
     (ind)-> data.dataToScroll[ind]
     if(typeof(value)!=threeDimDat.type )  throw(DomainError(value, "supplied value should be of compatible type - $(threeDimDat.type )"))  end #if

     return threeToTwoDimm(threeDimDat.type,slice,data.dimensionToScroll,threeDimDat  ) |>
     (twoDimDat)-> modSlice!(twoDimDat,coords,value)
end#modifySliceFull!

```@doc
Return number of slices present in on slice data - takes into account slices dimensions
```
function getSlicesNumber(data::FullScrollableDat)::Int32
return Int32(size(data.dataToScroll[1].dat)[data.dimensionToScroll])
end#getSlicesNumber



```@doc
Based on DataToScrollDims it will enrich passed CalcDimsStruct texture width, height and  heightToWithRatio
based on data passed from DataToScrollDims
```
function getHeightToWidthRatio( calcDim::CalcDimsStruct ,dataToScrollDims::DataToScrollDims)::CalcDimsStruct
  toSelect= filter(it-> it!=dataToScrollDims.dimensionToScroll , [1,2,3] )# will be used to get texture width and height

  return setproperties(calcDim, (imageTextureWidth= dataToScrollDims.imageSize[toSelect[1]]
                              ,imageTextureHeight =dataToScrollDims.imageSize[toSelect[2]]
                              ,heightToWithRatio =  dataToScrollDims.voxelSize[toSelect[1]]/dataToScrollDims.voxelSize[toSelect[2]]
                              ,textTextureZeros= calcDim.textTextureZeros  
  ))
end#getHeightToWidthRatio



```@doc
Based on DataToScrollDims ,2 dim cartesian coordinate and  slice number it gives 3 dimensional coordinate of mouse position
```
function cartTwoToThree(dataToScrollDims::DataToScrollDims 
                           ,sliceNumber::Int
                           ,cartIn::CartesianIndex{2})::CartesianIndex{3}
  toSelect= filter(it-> it!=dataToScrollDims.dimensionToScroll , [1,2,3] )# will be used to get texture width and height
resArr= [1,1,1]

resArr[dataToScrollDims.dimensionToScroll]=Int64(sliceNumber)
resArr[toSelect[1]] = cartIn[1]
resArr[toSelect[2]] = cartIn[2]
  return CartesianIndex(resArr[1],resArr[2],resArr[3]  )
end#cartTwoToThree




```@doc
Given function and actor it passes the function to forUndoVector -
   in case the length of the vector is too big the last element woill be removed
```
function addToforUndoVector(actor::SyncActor{Any, ActorWithOpenGlObjects} 
                           ,fun)

   push!(actor.actor.forUndoVector,fun)

    if(length(actor.actor.forUndoVector) >actor.actor.maxLengthOfForUndoVector )
      popfirst!(actor.actor.forUndoVector)
    end  

end#addToforUndoVector


```@doc
utility function to create series of ThreeDimRawDat from list of tuples where
first entry is String and second entry is 3 dimensional array with data 
```
function getThreeDims(list )

  return map(tupl->ThreeDimRawDat{typeof(tupl[2][1])}(typeof(tupl[2][1]),tupl[1],tupl[2]) ,list)

end#getThreeDims


# parameter_type(x::TextureSpec) = parameter_type(typeof(x))



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
   
   



    return setproperties(calcDimStruct, (correCtedWindowQuadHeight= Int32(round(correCtedWindowQuadHeight))
                                        ,correCtedWindowQuadWidth= Int32(round(correCtedWindowQuadWidth))
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



end#StructsManag


