

using DrWatson
@quickactivate "Medical segmentation evaluation"

module BasicStructs
using Parameters
```@doc
constants associated with image over which we will evaluate segmentations
```
@with_kw struct ImageConstants
    mvspx ::double # Voxelspacing x 
    mvspy::double # Voxelspacing y
    mvspz::double #mean Voxelspacing z
    isZConst::bool # true if slices thickness is the same in all image
    ZPositions::Vector{Double} # array of true physical positions (in mm) of slices relative to the begining - used in case we have variable thickness of slices
    numberOfVox::Int64 # number of voxels in image
end #ImageConstants
```@doc
configuration struct that when passed will marks what kind of metrics we are intrested in 
    
    ```
@with_kw struct ConfigurtationStruct
    anyFuzzy::Bool = false# is any of the metric calculated fuzzy
    

end #ConfigurtationStruct


```@doc
Struct holding all resulting metrics - if some metric was not calculated its value is just -1  
```
@with_kw struct ResultMetrics
    dice::Float64 = -1.0 #dice coefficient
    jaccard::Float64 =  -1.0 #jaccard coefficient
    gce::Float64 =  -1.0 #global consistency error
    vol::Float64 =  -1.0 # Volume metric
    randInd::Float64 = -1.0 # Rand Index 
    mi::Float64 = -1.0 # mutual information
    ic::Float64 = -1.0 # interclass correlation
    Kc::Float64 = -1.0 # Kohen Cappa
    Mahalanobis::Float64 = -1.0 # Mahalanobis distance
    Hausdorff::Float64 = -1.0 # Hausdorff distance

end #ResultMetrics






end#BasicStructs
