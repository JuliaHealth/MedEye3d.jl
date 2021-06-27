

using DrWatson
@quickactivate "Probabilistic medical segmentation"

module GaussianPure

using Base: Number
using Documenter
using Main.imageViewerHelper
using Statistics
using LinearAlgebra
using StaticArrays
```@doc
2. By iteratively  searching through the mask M array cartesian coordinates of all entries with value 7 will be returned.
```
function getCoordinatesOfMarkings(::Type{ImageNumb}, ::Type{maskNumb}, M::Array{maskNumb, 3}, I::Array{ImageNumb, 3} )  ::Vector{CartesianIndex{3}} where{ImageNumb,maskNumb}
    return filter((index)->M[index]==7 ,CartesianIndices(M))
end    

```@doc    
3.Given two cartesian coordinates it will calculate sum of absolute values of diffrences between x,y and z coordinates of both points
getOneNormDist\(pointA,pointB\)
```
function getOneNormDist(pointA::CartesianIndex{3},pointB::CartesianIndex{3}) ::Int
   return Main.imageViewerHelper.cartesianTolinear(pointB-pointA  ) 
end

```@doc
4. Now we will define the patch where we have set of coordinates q surrounding coordinate i
 in distance not bigger then z
```
function getCartesianAroundPoint(point::CartesianIndex{3},z ::Int)  ::Vector{CartesianIndex{3}}
    return Main.imageViewerHelper.cartesianCoordAroundPoint(point,z)
end    

```@doc
5.We need to define the patch Ω using getCartesianAroundPoint around each seed point - we will list of coordinates set  
markings - calculated  earlier in getCoordinatesOfMarkings  z is the size of the patch - it is one of the hyperparameters
return the patch of pixels around each marked point
```
function getPatchAroundMarks(markings ::Vector{CartesianIndex{3}}, z::Int) ::Vector{Vector{CartesianIndex{3}}}
    return [getCartesianAroundPoint(x,z) for x in markings]
end    
```@doc
6.Now we apply analogical operation to each point coordinates of each patch  to get set of sets of sets where the nested sub patch will be referred to as Ω_ij
markingsPatches is just the output of getPatchAroundMarks 
z is the size of the patch - it is one of the hyperparameters
return nested patches so we have patch around each voxel from primary patch
```
function allNeededCoord(markingsPatches ::Vector{Vector{CartesianIndex{3}}},z::Int ) ::Vector{Vector{Vector{CartesianIndex{3}}}}
    return [getPatchAroundMarks(x,z) for x in markingsPatches]
end    

```@doc
7.We define function that give set of cartesian coordinates  returns the vector where first entry is a sample mean and second one sample standard deviation 
 of values in image I in given coordinates
 first type is specyfing the type of number in image array second in the output - so we can controll what type of float it would be
getSampleMeanAndStd\(points,I\)
```
function  getSampleMeanAndStd(a ::Type{Numb},b ::Type{myFloat}, coords::Vector{CartesianIndex{3}} , I ::Array{Numb, 3} ) ::Vector{myFloat} where{Numb, myFloat}
    arr= I[coords]
    return [mean(arr), std(arr)]   
end

```@doc
8.Next we reduce each of the sub patch omega using getSampleMeanAndStd function and store result in patchStats
calculatePatchStatistics\(allNeededCoord,I\)
```
function calculatePatchStatistics(a ::Type{Numb},b ::Type{myFloat},allNeededCoord ::Vector{Vector{Vector{CartesianIndex{3}}}},I ::Array{Numb, 3}) ::Vector{Vector{Vector{myFloat}}}  where{Numb, myFloat}
    return [ [getSampleMeanAndStd(a,b, x,I) for x in outer ] for outer in  allNeededCoord]
end


```@doc
9.We calculate feature vector related to a seed  point  where we will normalize means and standard deviations
 from all pixels in a primary patch where each feature vector ba
```
function calculateFeatureVector(a ::Type{myFloat},patchStat ::Vector{Vector{myFloat}}) ::Vector{myFloat} where{ myFloat}
     return  [ getSumOverNorm(map(x->x[1], patchStat)) ,   getSumOverNorm(map(x->x[2], patchStat)) ] 
end    
```@doc
given vector with float values it divides sum of this vector  by norm of this vector
```
function getSumOverNorm(vect::Vector{myFloat}) ::myFloat  where{ myFloat}
   return sum(vect)/norm(vect,2)
end


```@doc
11.
Calculating the Covariance matrix for single 2 dimensional matrix 
patchStat means and standard deviations related to given seedpoint
```
function getCovarianceMatrix(a ::Type{myFloat},patchStat ::Vector{Vector{myFloat}}) ::SMatrix{2, 2, myFloat, 4}  where{ myFloat}
    means= [x[1] for x in patchStat]
    stds = [x[2] for x in patchStat]
    covv = cov(means,stds)
    return SMatrix{2}(var(means),covv,covv, var(stds))
    
end

```@doc
12.We calculate log of  normalizing constant for each seed point 
covarianceMatrix  is 2 by 2
fetureVectLength tells us about dimensionality of features
return calculated log of multivariate normal distribution
```
function getLogNormalConst(a ::Type{myFloat},covarianceMatrix ::SMatrix{2, 2, myFloat, 4}, fetureVectLength ::Int) :: myFloat where{ myFloat}
    return  -(fetureVectLength*  log(2π)+logdet(covarianceMatrix))/2
end    

```@doc
For convinience we will fuse here a step of creating feature ectors and covariance matrices
patchStat means and standard deviations related to given seedpoint
```
function getCovarianceMatricisAndFeatureVectors(a ::Type{myFloat},patchStats ::Vector{Vector{Vector{myFloat}}}) ::Vector{Tuple{SMatrix{2, 2, myFloat, 4}, Vector{myFloat}}}  where{ myFloat}
  return [(getCovarianceMatrix(a,patchStat ),calculateFeatureVector(a,patchStat)) for patchStat in patchStats ]
    
end

```@doc
13. We collect statistics associated with all seed points we will 
get resultant  vector of normalizing constant mean
 , covariance matrix and we additionaly calculate covariance matrix inverse
 All of those values will be then used in kernel to calculate a pdf \(probability density function\)
 M - mask 3 dimensional array
 I - Image 3 dimensional array
 floatType - points to precision with which we want to calculate generally best works with Float64
 imageTypeNumb - what is a type of numbers that constitues image data
 maskTypeNumb - what is a type of numbers that constitues mask data


 HyperParameters
 z- size of the radius of the patch \(1 norm radius\)


 Return the constants quadriple needed to efficiently calculate gaussians pdfs defined around seed points
 1. mean vector \( feature vector minus its mean\)
 2. covariance matrix inverse
 3. log of normalization constant
  ```
function getConstantsForPDF(floatType ::Type{myFloat},imageTyp ::Type{imageTypeNumb},maskType ::Type{maskTypeNumb} ,M::Array{maskTypeNumb, 3}, I::Array{imageTypeNumb, 3}, z::Int)   where{myFloat,imageTypeNumb,maskTypeNumb }

return getCoordinatesOfMarkings(imageTypeNumb,maskTypeNumb, M,I) |>
(seedsCoords) ->getPatchAroundMarks(seedsCoords,z ) |>
(patchCoords) ->allNeededCoord(patchCoords,z ) |>
(allCoords) ->calculatePatchStatistics(imageTyp, floatType, allCoords, I)|>
(patchStats) ->getCovarianceMatricisAndFeatureVectors(imageTyp, patchStats)|>
(fvSandCovs) ->fromFeatureVectorCalculateConstants(floatType,fvSandCovs )
end

```@doc
Given covariance matrix and feature vectors tuple calculates needed statistics for MV normal distribution

Return the constants quadriple needed to efficiently calculate gaussians pdfs defined around seed points
    1. mean vector \( feature vector minus its mean\)
    2. covariance matrix inverse
    3. log of normalization constant
    4.covariance matrix
  ```
function fromFeatureVectorCalculateConstants(floatType ::Type{myFloat}, fvSandCovs ::Vector{Tuple{SMatrix{2, 2, myFloat, 4}, Vector{myFloat}}})  where{myFloat }
    return [(fvSandCov[2],# mean
           inv(fvSandCov[1]), # just 4 numbers no point in cholesky
           getLogNormalConst(floatType,fvSandCov[1],2)# calculating the log of normalizing constant
           ,fvSandCov[1]
           ) for fvSandCov in  fvSandCovs]
   end

end


