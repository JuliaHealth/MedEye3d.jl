using LinearAlgebra: convert, Matrix
using BenchmarkTools: maximum
using Base: Flatten
using DrWatson
@quickactivate "Probabilistic medical segmentation"
dirToImageHelper = DrWatson.scriptsdir("MedImageTransformations","gaussianFromSeeds","GaussiansFromSeeds.jl")
include(dirToImageHelper)
using Main.GaussianPure
using Test
using BenchmarkTools
using Statistics
using LinearAlgebra
using Distributions
using PDMats
using StaticArrays


```@doc
2\)Example 3 dimensional matrix I with example marking mask M 
will be saved with known number of marked points  x on known position
We will assert thag wefound correctly those points and that 
```
@testset " getCoordinatesOfMarkings " begin 
A = ones(4,4,4)
B = ones(4,4,4)
coords = [CartesianIndex(1,2,3),CartesianIndex(1,4,3) ] 
A[coords].=7
@test Main.GaussianPure.getCoordinatesOfMarkings(Float64,Float64,A,B) ==coords
end # getCoordinatesOfMarkings


```@doc
3. We will test getOneNormDist\(pointA,pointB\)
 by manually supplying some points with know 1 norm distance for example
Assert equal 
```

@testset "getOneNormDist" begin 
    @test    Main.GaussianPure.getOneNormDist(CartesianIndex(1,1,1),CartesianIndex(2,2,2)) == 3

end # getOneNormDist
    

```@doc
4.Testing getCartesianAroundPoint\(point,z\)
```
@testset "getCartesianAroundPoint" begin 
    @test   Set(Main.GaussianPure.getCartesianAroundPoint(CartesianIndex(1,1,1),1)) == Set([CartesianIndex(1,1,1),CartesianIndex(0,1,1),
    CartesianIndex(2,1,1),CartesianIndex(1,0,1),CartesianIndex(1,2,1),CartesianIndex(1,1,0),CartesianIndex(1,1,2)])
end 
    

```@doc
5\)Testing primPatch 
a\)Length of primPatch  should be the same as length of SeedPoints,
b\)Given some point in SeedPoints is \(1,1,1\) should pass the test 4
```
@testset "getPatchAroundMarks" begin 
    A = ones(4,4,4)
    B = ones(4,4,4)
    coords = [CartesianIndex(1,2,3),CartesianIndex(1,4,3) ,CartesianIndex(1,1,1) ] 
    A[coords].=7
    @test size(Main.GaussianPure.getPatchAroundMarks( coords,1))[1] ==3
    toTes = Main.GaussianPure.getPatchAroundMarks(coords,1)


    @test   Set(toTes[3]) == Set([CartesianIndex(1,1,1),CartesianIndex(0,1,1),
    CartesianIndex(2,1,1),CartesianIndex(1,0,1),CartesianIndex(1,2,1),CartesianIndex(1,1,0),CartesianIndex(1,1,2)])
end


```@doc
6\)Testing allNeededCoord
a\)Length of allNeededCoord should be the same as length of SeedPoints,
b\) size of each subentry given z=1 should be the same as in test 4 
c\)Given some point in sub sub entry is \(1,1,1\) should pass the test 4
```
@testset "allNeededCoord" begin 
    C =  [[CartesianIndex(1, 1, 1), CartesianIndex(1, 1, 3), CartesianIndex(0, 2, 3), CartesianIndex(1, 2, 3), CartesianIndex(2, 2, 3), CartesianIndex(1, 3, 3), CartesianIndex(1, 2, 4)]
    ,[CartesianIndex(1, 4, 2), CartesianIndex(1, 3, 3), CartesianIndex(0, 4, 3), CartesianIndex(1, 4, 3), CartesianIndex(2, 4, 3), CartesianIndex(1, 5, 3), CartesianIndex(1, 4, 4)]
    ,[CartesianIndex(1, 1, 0), CartesianIndex(1, 0, 1), CartesianIndex(0, 1, 1), CartesianIndex(1, 1, 1), CartesianIndex(2, 1, 1), CartesianIndex(1, 2, 1), CartesianIndex(1, 1, 2)]
    ]
    toTes= Main.GaussianPure.allNeededCoord(C,1)
    @test   Set(toTes[1][1]) == Set([CartesianIndex(1,1,1),CartesianIndex(0,1,1),
    CartesianIndex(2,1,1),CartesianIndex(1,0,1),CartesianIndex(1,2,1),CartesianIndex(1,1,0),CartesianIndex(1,1,2)])

end
```@doc
7\)
We supply 3 dimensional matrix of ones  as I and supply all of the coordinates we should get back the mean =1 and std = 0
```
@testset "getSampleMeanAndStd" begin 
    A = ones(4,4,4)

    coords = [CartesianIndex(1,2,3),CartesianIndex(1,4,3) ,CartesianIndex(1,1,1) ] 
    @test Main.GaussianPure.getSampleMeanAndStd(Float64, Float64 , coords, A) == [1,0]
end
```@doc
 
8.We supply set of different coordinates to 3 dimensional array of ones we should get back two 
dimensional list where each sublist should have first entry 1 and second 0
```
@testset "calculatePatchStatistics" begin 
    coordsList = Main.GaussianPure.allNeededCoord(Main.GaussianPure.getPatchAroundMarks(Main.GaussianPure.getCartesianAroundPoint(CartesianIndex(4,4,4),1),1),1)
    I = ones(9,9,9)
    res = Main.GaussianPure.calculatePatchStatistics(Float64, Float64 , coordsList, I)
    @test res[1][1][1] ==1
    @test res[1][2][2] ==0
end



```@doc
we are just checking weather all ot previous functions work as expected 
``` 
@testset "miniIntegrationTest" begin 
    using LinearAlgebra
    using Combinatorics
  
      list = Main.GaussianPure.allNeededCoord(Main.GaussianPure.getPatchAroundMarks(Main.GaussianPure.getCartesianAroundPoint(CartesianIndex(1,1,1),1),1),1)
     Flattened = reduce(vcat, reduce(vcat,list) )
     function oneNormDisttt(a,b)
      return Main.GaussianPure.getOneNormDist(a,b)
     end
  
  function zip_collect(itr)
      itrsize = Base.IteratorSize(itr)
      itrsize isa Base.HasShape && (itrsize = Base.HasLength())
      Base._collect(1:1, itr, Base.IteratorEltype(itr), itrsize)
  end
  
  comb = combinations(Flattened,2)
  
  dists = [oneNormDisttt(x[1],x[2]) for x in comb]
  
  @test maximum(dists) == 6
  @test minimum(dists) ==0
  
  end



```@doc
calculateFeatureVectors\(patchStats\) - 
if we will keep all matrices as matrices of ones after reduction 
    and norm we still should have get mean of 1 and std of 0
```
@testset "calculateFeatureVector" begin 
    coordsList = Main.GaussianPure.allNeededCoord(Main.GaussianPure.getPatchAroundMarks( [CartesianIndex(4,4,4),CartesianIndex(4,3,4) ,CartesianIndex(3,3,3) ] ,1),1)
    I = ones(9,9,9)
    I[CartesianIndex(4,4,4)]=7
    maximum(I)
    patchStats = Main.GaussianPure.calculatePatchStatistics(Float64, Float64 , coordsList, I)
    patchStats[1][1]
    @test Main.GaussianPure.calculateFeatureVector(Float64, [  [1.,1], [1.,1.],[1.,1.],[1.,1.] ] ) ==[2,2]


    @test  size(patchStats[1])[1] == 7

end



```@doc
11\)If we will supply vectors of ones the covariance matrix should have all entries =0
```
@testset "getCovarianceMatrix" begin 
    coordsList = Main.GaussianPure.allNeededCoord(Main.GaussianPure.getPatchAroundMarks( [CartesianIndex(4,4,4),CartesianIndex(4,3,4) ,CartesianIndex(3,3,3) ] ,1),1)
    I = ones(9,9,9)
    I[CartesianIndex(4,4,4)]=7
    maximum(I)
    patchStats = Main.GaussianPure.calculatePatchStatistics(Float64, Float64 , coordsList, I)
    patchStats[1]
    @test Main.GaussianPure.getCovarianceMatrix(Float64,[[1.,1.,1.],[1.,1.,1.]  ]) ==SMatrix{2}(0,0,0,0)
    
end



```@doc
12\)
First we will compare the normalizing constant to one calculated by the Distributions.jl

```
@testset "getCovarianceMatrices" begin 
# mvnormal_c0(g::AbstractMvNormal) = -(length(g) * convert(eltype(g), log2π) + logdet(d.Σ))/2
covA = ones(2,2)
covA[1,1]=5
covA[2,2]=5
covB = SMatrix{2}(5.,1.,1.,5.) 
mu = [3.,3.]
mvNorm = Distributions.MvNormal(mu,covA)
@test  Main.GaussianPure.getLogNormalConst(Float64, covB,2) == Distributions.mvnormal_c0(mvNorm)
 

end




@testset "getCovarianceMatricisAndFeatureVectors" begin 
    @test Main.GaussianPure.getCovarianceMatricisAndFeatureVectors(Float64,[[[1.,1.,1.],[1.,1.,1.]  ]])[1][1] ==SMatrix{2}(0,0,0,0)
    @test Main.GaussianPure.getCovarianceMatricisAndFeatureVectors(Float64, [[  [1.,1], [1.,1.],[1.,1.],[1.,1.] ]] )[1][2] ==[2,2]
end



```@doc
13\)We apply tests from 12 to each entry
```

```@doc
14.We supply to our  normal distributions some mean and covariance and we evaluate pdf on some entries 
- and compare to the results of multivariate gaussian from Distributions.jl package

also 


We will supply as a feature vector vector filled with randomly generated numbers 
of given dimensionality on the basis of which we will calculate normalizing constant 
- next in a loop we will randomly crate x vector of appropriate dimensionality and we
 will evaluate gaussian pdf with calculated normalizing constant test will be passed
  if calculated values will be always between 0 and 1

We also need to establish waether we would also get positive semidefinite covariance matrix as we should
```

    @testset "fromFeatureVectorCalculateConstants" begin 
 
function getExampleMvGaussianDats(M,I,coords)
    M[coords].=7
    z = 1
    res = Main.GaussianPure.getConstantsForPDF(Float64,Float64,Float64,M,I ,z)[1]
    covv =  res[4]
    covA = ones(2,2)
    covA[1,1]=covv[1,1]
    covA[2,2]=covv[2,2]
    covA[2,1]=covv[2,1]
    covA[1,2]=covv[1,2]
    mu =res[1]
    mvNorm = Distributions.MvNormal(mu,covA)
    return (res, mvNorm)
end



function getPdfEvals(xxx,dists)
        


    distData = dists[1]
    mvNormm = dists[2]


    distProb = Distributions.pdf(mvNormm, xxx) # we check the probability in mean ... so we know it needs to be high

    centered =  xxx.-distData[1]
    myProb = exp(distData[3]- (transpose( centered )*distData[2]* centered )/2 )

    return(distProb,myProb)
end


M = rand(15,15,15)
I = rand(15,15,15)
coords = [CartesianIndex(5,5,5),CartesianIndex(10,10,10) ] 

dists = getExampleMvGaussianDats(M,I,coords)
dists[2]
distData = dists[1]
xxx = distData[1].+0.1

evals=getPdfEvals(xxx,dists)
@test evals[1]≈evals[2]


## now we will create and "train "


d1 = Binomial(300)
d2 = Normal(50.0,15.0 )

I1 = rand(d1,15,15,15)
I2 = rand(d2,15,15,15)


distsLow = getExampleMvGaussianDats(M,I1,coords)
distsHigh = getExampleMvGaussianDats(M,I2,coords)

distLowData = distsLow[1]
xxxB = distLowData[1].+0.1
xxxC = distsHigh[1][1].+0.1

#checking is it giving higher probability on 
@test getPdfEvals(xxxB,distsLow)[1] > getPdfEvals(xxxB,distsHigh)[1]

randLow = rand(d1,2)
randHigh = rand(d2,2)


#@test getPdfEvals(randLow,distsLow)[1]>  getPdfEvals(randLow,distsHigh)[1] ||    getPdfEvals(randLow,distsLow)[1]≈ getPdfEvals(randLow,distsHigh)[1]  
@test getPdfEvals(randHigh,distsLow)[1] <= getPdfEvals(randHigh,distsHigh)[1] ||    getPdfEvals(randHigh,distsLow)[1] ≈ getPdfEvals(randHigh,distsHigh)[1] 

distsLow[2]
distsHigh[2]

end