using DrWatson
@quickactivate "Probabilistic medical segmentation"
"""
utilities for dealing data structs like FullScrollableDat or SingleSliceDat
"""
module StructsManag
using Main.DataStructs
export threeToTwoDimm,modSlice!, threeToTwoDimm,modifySliceFull!,getSlicesNumber

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
   data.dat
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
                ,slice::Int
                ,sliceDim::Int
                ,threedimDat::ThreeDimRawDat{T})::TwoDimRawDat{T} where {T}
                arr=[":",":",":"]
                arr[sliceDim]="$slice"
               return TwoDimRawDat{T}(typ,threedimDat.name,view(threedimDat.dat,eval(Meta.parse(arr[1])),eval(Meta.parse(arr[2])),eval(Meta.parse(arr[3]))   )   )
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
function getSlicesNumber(data::FullScrollableDat)::Int
return size(data.dataToScroll[1].dat)[data.dimensionToScroll]
end#getSlicesNumber

end#StructsManag

