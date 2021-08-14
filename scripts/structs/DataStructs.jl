using DrWatson
@quickactivate "Probabilistic medical segmentation"
"""
structs helping managing and storing data
"""
module DataStructs
using Parameters, Main.BasicStructs, Dictionaries
export  RawDataToDisp,TwoDimRawDat, ThreeDimRawDat, DataToDisp,FullScrollableDat,SingleSliceDat

```@doc
hold raw Data that can be send to be displayed 
```
abstract type RawDataToDisp end

```@doc
2 dimensional ata for displaying single slice
struct is mutable becouse in case of the masks data can be changed multiple times and rapidly   
```
@with_kw mutable struct TwoDimRawDat{T} <: RawDataToDisp
   type::Type{T}= UInt8# easy access to type
   name::String=""#associated name
   dat::AbstractArray{T, 2}=ones(type,2,2)# raw pixel data
end#2DimRawDat


```@doc
3 dimensional data for displaying single slice
struct is mutable becouse in case of the masks data can be changed multiple times and rapidly   
```
@with_kw mutable struct ThreeDimRawDat{T} <: RawDataToDisp
   type::Type{T}= UInt8# easy access to type
   name::String=""#associated name
   dat::AbstractArray{T, 3}=ones(type,2,2,2)# raw voxel data
end#2DimRawDat

```@doc
given Vector of tuples where first is string and second is RawDataToDisp
it creates dictionary where keys are those strings - names and values are indicies where they are found
```
function getLocationDict(listt)::Dictionary{String, Int64}
   return Dictionary(map(it->it.name,listt),collect(eachindex(listt)))
    
end#getLocationDict




```@doc
hold Data that can be send to be displayed with required metadata
```
abstract type DataToDisp end


FullScrollableDatStr="""
Data that can be displayed and scrolled (so we have multiple slices)
struct is mutable becouse in case of the masks data can be changed multiple times and rapidly
"""
@doc FullScrollableDatStr
@with_kw mutable struct FullScrollableDat<: DataToDisp
    dimensionToScroll::Int= 3 # by which dimension we should scroll so for example if set to 3 one and we have slice number x we will get data A by A[:,:,x] if dimensionToScroll = 2 ->A[:,x,:]...
    dataToScroll::Vector{ThreeDimRawDat}=[ThreeDimRawDat()] # tuples where first entry is name of image that we given in configuration, and second entry is data that we want to pass
# data to display in form of a list Of tuples where first entry will be used as headtitle for the data that is an value ;second entry -  value is a vector where each entry will be displayed in separate line
    mainTextToDisp::Tuple{String, Vector{String}} = ("",[]) # text that will be displayd for all data (scrolling will not affect it)
    sliceTextToDisp::Vector{Tuple{String, Vector{String}}}=[] # text that will be associated with given slice - length of this needs to be the same as  number of slices we want to scroll through
    #all metrics that were not measured and are possible in ResultMetrics struct will have associated value = -1
    segmMetr::ResultMetrics=ResultMetrics() #results of metrics for whole 3d image
    segmMetrs::Vector{ResultMetrics}=[] #results of metrics for each slice - array needs to be of the same size as number of slices in passed data
    nameIndexes::Dictionary{String, Int64}= getLocationDict(dataToScroll)  #gives a way of efficient querying by supplying dictionary where key is a name we are intrested in and a key is index where it is located in our array
end #fullScrollableDat

```@doc
Data for displaying single slice
struct is mutable becouse in case of the masks data can be changed multiple times and rapidly   
```
@with_kw mutable struct SingleSliceDat<: DataToDisp
    listOfDataAndImageNames::Vector{TwoDimRawDat}=[TwoDimRawDat()]   # tuples where first entry is name of image that we given in configuration, and second entry is data that we want to pass
    textToDisp::Tuple{String, Vector{String}} =  ("",[])# data to display in form of a list Of tuples where first entry will be used as headtitle for the data that is an value ;second entry -  value is a vector where each entry will be displayed in separate line
    segmMetr::ResultMetrics=ResultMetrics() #results metrics associated with this slice 
    nameIndexes::Dictionary{String, Int64}= getLocationDict(listOfDataAndImageNames)  #gives a way of efficient querying by supplying dictionary where key is a name we are intrested in and a key is index where it is located in our array
    sliceNumber::Int=1 # if we want it to be tamporarly  associated with some slice in scrollable data
end #fullScrollableDat





end# DataStructs

