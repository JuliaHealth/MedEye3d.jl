

using DrWatson
@quickactivate "Probabilistic medical segmentation"

```@doc
Main goal is to develop on the basis of some data of pixels in the image paraeters of multivariate gaussian distribution ; also calculate all required constants in order to 
pi- voxel seed point where i∈{1..k}
Ωi- set of pixels defined around some voxel i includes all pixels which max-imum  euclidean  distance  from  any  point  of  voxel  i  is  smaller  than  z  and  z  isdefined in units where one unit is a pixel width
    Ωi={q|dist(i,q)< z}where q will describe each voxel in the patch Ωi
Ni-multivariate gaussian distribution defined over Ωiwherei∈{1..k}
Fq- feature vector containing the mean and standard deviation associatedwith each voxel in a Ωiwhere those statistics were calculated for separete patcheswhere each voxel was in center

1. Using GUI set  the points seeds on image I  that are placed inside given organ -liver, this information will be stored in mask array of the same dimensionality as I -in this implementation the value assigned in mask to the place user markedis set arbitrary to 7).

2. By iteratively  searching through the mask M array cartesian coordinates of all entries with value 7 will be returned.
getMarkings(M, I)
SeedPoints= [] (empty array)
For coord in cartesianCoordinates(I)
	el = I[coord]
	If el==7
Out add(el)
	end if
end for

3.Given two cartesian coordinates it will calculate sum of absolute values of diffrences between x,y and z coordinates of both points
getOneNormDist(pointA,pointB)
Return |pointA.x -pointB.x |+ |pointA.y -pointB.y | +|pointA.z -pointB.z |

4. Now we will define the patch Ωi={q|getOneNormDist(i,q)<= z} where we have set of coordinates q surrounding coordinate i in distance not bigger then z
getCartesianAroundPoint(point,z)  
return set {q|getOneNormDist(i,q)<= z}

5.We need to define the patch Ω using getCartesianAroundPoint around each seed point - we will list of coordinates set  
primPatches = SeedPoints.map(x-> getCartesianAroundPoint(x,z))

6.Now we apply analogical operation to each point coordinates of each patch $Ω_i$ to get set of sets of sets where the nested sub patch will be referred to as $Ω_{ij}$
allNeededCoord = primPatches .map(point -> getCartesianAroundPoint(point,z)  )

7.We define function that give set of cartesian coordinates  returns the vector where first entry is a sample mean and second one sample standard deviation  of values in image I in given coordinates
getSampleMeanAndStd(points,I) 
values = Points.map (point-> I[point])
return[mean(values ), std(values )]

8.Next we reduce each of the sub patch  $Ω_{ij}$ using getSampleMeanAndStd function and store result in patchStats
calculatePatchStatistics(allNeededCoord,I)
allNeededCoord.map($Ω_{i}$ ->   $Ω_{i}$.map($Ω_{ij}$  -> getSampleMeanAndStd (Ω_{ij},I)      ))

9.We calculate feature vector related to each seed point  where we will normalize means and standard deviations from all pixels in a primary patch where each feature vector is defined as in equation below
$F_p = [ \frac{ \sum_{i \in \Omega_p } { \mu_i }} {||\Omega||},\frac{ \sum_{i \in \Omega_p } { \sigma_i }} {||\Omega||} ]^T $
calculateFeatureVectors(patchStats)
Out = []
For (subPatchStats,index)- in patchStats
patchNorm =  TwoNorm(primPatches[index])
Out  add subPatchStats/patchNorm   # element wise and division
End for
Return Out 

10.We calculate mean two dimensional array by subtracting from feature vector f its mean for each seed point
$\mu_i = f- \bar{f}$

11.We calculate covariance (2x2) matrix for each seed point
$\cov_i $ = [ [ var(f1),cov(f1,f2) ],[cov(f2,f1),  var(f2) ]  ] - 2 dimensional lists f1 is a vector of first entries and f2 vector of second entries
We also pre calculate inverse of the covariance matrix
$invSigma = inv(\Sigma) $

12.We calculate normalizing constant for each seed point 
logNorC = $log(\frac{1}{2\pi^{\frac{k}{2}} | \Sigma |^{1/2}})$

13. We apply  step 10,11,12 to all of the statistics associated with all seed points we will get resultant  vector of normalizing constant mean , covariance matrix and its inverse

14. We create set of lambda functions  that will calculate the multivariate random gaussian in log space given vector x
Lambda (x)-> llogNorC+ $-\frac{1}{2} *  \mu^T *  invSigma * \mu  $
```
module GaussianPure
using Base: Number
using Documenter


```@doc
2. By iteratively  searching through the mask M array cartesian coordinates of all entries with value 7 will be returned.
getMarkings\(M, I\)
SeedPoints= \[\] \(empty array\)
For coord in cartesianCoordinates\(I\)
	el \= I\[coord\]
	If el\=\=7
Out add\(el\)
	end if
end for
```
function getCoordinatesOfMarkings(M::Array{Number, 3}, I::Array{Number, 3} ) ::Vector{CartesianIndex{3}} 
    filter((index)->I[index]==7 ,CartesianIndices(M))
end    



```@doc    
3.Given two cartesian coordinates it will calculate sum of absolute values of diffrences between x,y and z coordinates of both points
getOneNormDist(pointA,pointB)
Return |pointA.x -pointB.x |+ |pointA.y -pointB.y | +|pointA.z -pointB.z |
```



```@doc
4. Now we will define the patch Ωi={q|getOneNormDist(i,q)<= z} where we have set of coordinates q surrounding coordinate i in distance not bigger then z
getCartesianAroundPoint(point,z)  
return set {q|getOneNormDist(i,q)<= z}
```


```@doc
5.We need to define the patch Ω using getCartesianAroundPoint around each seed point - we will list of coordinates set  
primPatches = SeedPoints.map(x-> getCartesianAroundPoint(x,z))
```



```@doc
6.Now we apply analogical operation to each point coordinates of each patch $Ω_i$ to get set of sets of sets where the nested sub patch will be referred to as $Ω_{ij}$
allNeededCoord = primPatches .map(point -> getCartesianAroundPoint(point,z)  )
```


```@doc
7.We define function that give set of cartesian coordinates  returns the vector where first entry is a sample mean and second one sample standard deviation  of values in image I in given coordinates
getSampleMeanAndStd(points,I) 
values = Points.map (point-> I[point])
return[mean(values ), std(values )]
```


```@doc
8.Next we reduce each of the sub patch  $Ω_{ij}$ using getSampleMeanAndStd function and store result in patchStats
calculatePatchStatistics(allNeededCoord,I)
allNeededCoord.map($Ω_{i}$ ->   $Ω_{i}$.map($Ω_{ij}$  -> getSampleMeanAndStd (Ω_{ij},I)      ))
```


```@doc
9.We calculate feature vector related to each seed point  where we will normalize means and standard deviations from all pixels in a primary patch where each feature vector is defined as in equation below
$F_p = [ \frac{ \sum_{i \in \Omega_p } { \mu_i }} {||\Omega||},\frac{ \sum_{i \in \Omega_p } { \sigma_i }} {||\Omega||} ]^T $
calculateFeatureVectors(patchStats)
Out = []
For (subPatchStats,index)- in patchStats
patchNorm =  TwoNorm(primPatches[index])
Out  add subPatchStats/patchNorm   # element wise and division
End for
Return Out 
```


```@doc
10.We calculate mean two dimensional array by subtracting from feature vector f its mean for each seed point
$\mu_i = f- \bar{f}$
```


```@doc
11.We calculate covariance (2x2) matrix for each seed point
$\cov_i $ = [ [ var(f1),cov(f1,f2) ],[cov(f2,f1),  var(f2) ]  ] - 2 dimensional lists f1 is a vector of first entries and f2 vector of second entries
We also pre calculate inverse of the covariance matrix
$invSigma = inv(\Sigma) $
```


```@doc
12.We calculate normalizing constant for each seed point 
logNorC = $log(\frac{1}{2\pi^{\frac{k}{2}} | \Sigma |^{1/2}})$
```


```@doc
13. We apply  step 10,11,12 to all of the statistics associated with all seed points we will get resultant  vector of normalizing constant mean , covariance matrix and its inverse
```

```@doc
14. We create set of lambda functions  that will calculate the multivariate random gaussian in log space given vector x
Lambda (x)-> llogNorC+ $-\frac{1}{2} *  \mu^T *  invSigma * \mu  $
```









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

