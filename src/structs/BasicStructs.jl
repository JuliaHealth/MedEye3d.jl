
module BasicStructs
using Parameters
export ImageConstants, ConfigurtationStruct, ResultMetrics
"""
constants associated with image over which we will evaluate segmentations
"""
@with_kw struct ImageConstants
    mvspx ::Float64 # Voxelspacing x
    mvspy::Float64 # Voxelspacing y
    mvspz::Float64 #mean Voxelspacing z
    isZConst::Bool # true if slices thickness is the same in all image
    ZPositions::Vector{Float64} # array of true physical positions (in mm) of slices relative to the begining - used in case we have variable thickness of slices
    numberOfVox::Int64 # number of voxels in image
end #ImageConstants
"""
configuration struct that when passed will marks what kind of metrics we are intrested in

    """
@with_kw struct ConfigurationStruct
    anyFuzzy::Bool = false# is any of the metric calculated fuzzy


end #ConfigurtationStruct


"""
Struct holding all resulting metrics - if some metric was not calculated its value is just -1
"""
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
