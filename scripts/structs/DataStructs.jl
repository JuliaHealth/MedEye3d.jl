using DrWatson
@quickactivate "Probabilistic medical segmentation"
"""
structs helping managing and storing data
"""
module DataStructs
using Parameters,Dictionaries

```@doc
hold Data that can be send to be displayed
```
abstract type DataToDisp end


```@doc
Data that can be displayed and scrolled (so we have multiple slices)
struct is mutable becouse in case of the masks data can be changed multiple times and rapidly
```
@with_kw mutable struct FullScrollableDat<: DataToDisp
    dimensionToScroll::Int= 3 # by which dimension we should scroll so for example if set to 3 one and we have slice number x we will get data A by A[:,:,x] if dimensionToScroll = 2 ->A[:,x,:]...
    dataToScroll::Vector{Tuple{String, Array{T, 3} where T}} # tuples where first entry is name of image that we given in configuration, and second entry is data that we want to pass
# data to display in form of a list Of tuples where first entry will be used as headtitle for the data that is an value ;second entry -  value is a vector where each entry will be displayed in separate line
    mainTextToDisp::Tuple{String, Vector{String}} # text that will be displayd for all data (scrolling will not affect it)
    sliceTextToDisp::Vector{Tuple{String, Vector{String}}} # text that will be associated with given slice - length of this needs to be the same as  number of slices we want to scroll through
    #all metrics that were not measured and are possible in ResultMetrics struct will have associated value = -1
    segmMetr::ResultMetrics=ResultMetrics() #results of metrics for whole 3d image
    segmMetrs::Vactor{ResultMetrics}=[] #results of metrics for each slice - array needs to be of the same size as number of slices in passed data
end #fullScrollableDat

```@doc
Data for displaying single slice
struct is mutable becouse in case of the masks data can be changed multiple times and rapidly
    
```
@with_kw mutable struct SingleSliceDat<: DataToDisp
    listOfDataAndImageNames::Tuple{String, Array{T, 2} where T}   # tuples where first entry is name of image that we given in configuration, and second entry is data that we want to pass
    textToDisp::Tuple{String, Vector{String}} = ("".[])# data to display in form of a list Of tuples where first entry will be used as headtitle for the data that is an value ;second entry -  value is a vector where each entry will be displayed in separate line
    segmMetr::ResultMetrics=ResultMetrics() #results metrics associated with this slice 
end #fullScrollableDat




end# DataStructs