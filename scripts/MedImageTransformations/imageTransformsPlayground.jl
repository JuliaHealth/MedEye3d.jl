

getOrCreateMaskData(Int64, "liver", "trainingScans/liver-orig001", (10,10,10), RGBA(0,0,255,0.4))






exmpleH = @spawnat persistenceWorker Main.h5manag.getExample()
arrr= fetch(exmpleH)

minimumm = -1000

maximumm = 2000
imageDim = size(arrr)
maskF = @spawnat persistenceWorker Main.h5manag.getOrCreateMaskData(Int16, "liverOrganMask", "trainingScans/liver-orig001", imageDim, RGBA(0,0,255,0.4))
mask = fetch(maskF)

using Main.imageViewerHelper
using Main.MyImgeViewer


singleCtScanDisplay(arrr, [mask],minimumm, maximumm)


widerInd = unique ∘ collect ∘ Iterators.flatten ∘ map((c->Main.imageViewerHelper.cartesianCoordAroundPoint(c,2)),Main.imageViewerHelper.cartesianCoordAroundPoint(CartesianIndex(0,0,0),2) ) 

arra = mask.maskArrayObs[]


#this will find all the cartesian indexes related to place clicked
#this will find all the cartesian indexes related to pixels around clicked patch
#we connect those indexes
collectedIndexes = filter((ind) -> arra[ind] == 0.5, CartesianIndices(arra))
# now we need to separate patches we will do it by first sorting by sth like k means clustering so we will store in a list all of the indexes that are no more than 2 units from given index
#we use sets to increase speed of finding intersections  also sets are naturally unique  so we will not need additional operations
closeIndexes = map(((ind) ->Set(
                    filter( (innerind) -> 
                    abs(imageViewerHelper.cartesianTolinear(innerind)  - imageViewerHelper.cartesianTolinear(ind))<3
                         ,collectedIndexes)))
                    ,collectedIndexes)
# next we will concatenate all sets that have any common element
```@doc
we are checking weather any set in arr have some common element with setA if yes wefuse them if not we add setA to arr
setA- first set to check
arr - array of resulting sets
  ```
function ifIntersectAdd(arr, setA) 
if any intersect
end #ifIntersectAdd


closeIndexes = 
reduce(+, [], [1,2,3,4,5])  


intersect
# after we concateneded thos we will take only unique elements







# @spawnat persistenceWorker Main.h5manag.saveMaskData!(Int16, mask)

zz = @spawnat persistenceWorker Main.h5manag.saveMaskData!(Int16, mask)
fetch(zz)


exmpleH = getExample()
imageDim = size(exmpleH)
mask =getOrCreateMaskData(Int16, "liverOrganMask", "trainingScans/liver-orig001", imageDim, RGBA(0,0,255,0.4))
saveMaskDataC!(Int16, mask)













```@doc
set gaussian distribution using Distributions.jl 
it takes the 3 dimensional patch of points flattens it and on this basis we gat a distribution
```
function getGaussian(Ωi::Array) ::MvNormal
    flattened = collect(Iterators.flatten(Ωi))
    return MvNormal(mean(flattened),std(flattened) )
end
```@doc
creates an expression with precomputed all constant parts of the mulitivariate gaussian distribution so it later can be applied more efficiently
it will be based mostly on Distibutions.jl package implementation and nature article algorithm https://www.nature.com/articles/s41598-021-85436-7/tables/2

```
function preapareSingleGaussianPdF()


```@doc
normalizing constant copied from Distributions.jl
```
mvnormal_c0(g::AbstractMvNormal) = -(length(g) * convert(eltype(g), log2π) + logdet(d.Σ))/2







# taken from https://github.com/JuliaStats/Distributions.jl/blob/35125579a77b938f8d4a8bd2be23cbf1f2ddf225/src/multivariate/mvnormal.jl



###########################################################
#
#   Abstract base class for multivariate normal
#
#   Each subtype should provide the following methods:
#
#   - length(d):        vector dimension
#   - mean(d):          the mean vector (in full form)
#   - cov(d):           the covariance matrix (in full form)
#   - invcov(d):        inverse of covariance
#   - logdetcov(d):     log-determinant of covariance
#   - sqmahal(d, x):        Squared Mahalanobis distance to center
#   - sqmahal!(r, d, x):    Squared Mahalanobis distances
#   - gradlogpdf(d, x):     Gradient of logpdf w.r.t. x
#   - _rand!(d, x):         Sample random vector(s)
#
#   Other generic functions will be implemented on top
#   of these core functions.
#
###########################################################
using Distributions
rand(3,3,3)


struct MvNormalKnownCov{Cov<:AbstractPDMat}
    Σ::Cov
end

struct MvNormalKnownCovStats{Cov<:AbstractPDMat}
    invΣ::Cov              # inverse covariance
    sx::Vector{Float64}    # (weighted) sum of vectors
    tw::Float64            # sum of weights
end

function fit_mle(g::MvNormalKnownCov, x::AbstractMatrix{Float64}, w::AbstractVector)
    d = length(g)
    (size(x,1) == d && size(x,2) == length(w)) ||
        throw(DimensionMismatch("Inconsistent argument dimensions."))
    μ = BLAS.gemv('N', inv(sum(w)), x, vec(w))
    MvNormal(μ, g.Σ)
end

# + becouse we are in log space
mvnormal_c0(g::AbstractMvNormal) = -(length(g) * convert(eltype(g), log2π) + logdetcov(g))/2



_logpdf(d::AbstractMvNormal, x::AbstractVector) = mvnormal_c0(d) - sqmahal(d, x)/2

function _logpdf!(r::AbstractArray, d::AbstractMvNormal, x::AbstractMatrix)
    sqmahal!(r, d, x)
    c0 = mvnormal_c0(d)
    for i = 1:size(x, 2)
        @inbounds r[i] = c0 - r[i]/2
    end
    r
end

_pdf!(r::AbstractArray, d::AbstractMvNormal, x::AbstractMatrix) = exp!(_logpdf!(r, d, x))

```@doc
so we will calculate multivariate log normal
1) we need to calculate the squared  Mahalanobis distance what is a fancy way to define the exponent in the multivariate gaussian
2) we need to calculate normalization constant
3) as we are in log space we need to add those 2 thinks
```


#basically square mahalanoboius distance is what is in exponent of multivariate gaussian
sqmahal(d::AbstractMvNormal, x::AbstractMatrix) = sqmahal!(Vector{promote_type(partype(d), eltype(x))}(undef, size(x, 2)), d, x)

sqmahal(d::MvNormal, x::AbstractVector) = invquad(d.Σ, x .- d.μ)

#invquad! is a function of PDmats compute x' * inv(a) * x when `x` is a vector. 
# so this function takes r - which is just allocated memory ; and covariance and mena from our Distribution
# x is a parameter  which we take - we ask about probability of x
sqmahal!(r::AbstractVector, d::MvNormal, x::AbstractMatrix) =
    invquad!(r, d.Σ, x .- d.μ)





    








    using CUDA
    function gpu_add1!(y, x)
        for i = 1:length(y)
            @inbounds y[i] += x[i]
        end
        return nothing
    end
    
    fill!(y_d, 2)
    @cuda gpu_add1!(y_d, x_d)
    @test all(Array(y_d) .== 3.0f0)



using Latexify
print(latexify("x+y/(b-2)^2"))

