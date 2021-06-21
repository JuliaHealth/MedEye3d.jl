using DrWatson
@quickactivate "Probabilistic medical segmentation"
dirToImageHelper = DrWatson.scriptsdir("MedImageTransformations","gaussianFromSeeds","GaussiansFromSeeds.jl")
include(dirToImageHelper)
using Main.GaussianPure
using Test
```@doc
The points are set in the same order as in GaussianPure module
1)In algorithm  later on statistics will be calculated of the patches around points marked by user in order to be sure that all of the patch Ωp and patch around border pixels of this patch will be in the organ of choice in GUI the patch of the diameter $2* \Omega_i $ diameter will be shown for visual inspection whether all of the required voxels are present in the organ of choice.

2)Example image $I_i$ with example marking mask $M_i$ will be saved with known number of marked points  x on known slice zSlice
arr = getMarkings($M_i$,$I_i$) 
Assert equals length(arr) = x

For coord in arr 
	Assert equals z value of coord = zSlice
end for 

3)We will test getOneNormDist(pointA,pointB) by manually supplying some points with know 1 norm distance for example
Assert equal getOneNormDist(CartesianCoordinate(1,1,1),CartesianCoordinate(2,2,2)) == 3

4)Testing getCartesianAroundPoint(point,z) given coordinate (1,1,1) and distance 1 we should get in result set containing { (1,1,1),(0,1,1),(2,1,1),(1,0,1),(1,2,1),(1,1,0),(1,1,2) }

5)Testing primPatch 
a)Length of primPatch  should be the same as length of SeedPoints,
b) size of each entry given z=1 should be the same as in test 4 
c)Given some point in SeedPoints is (1,1,1) should pass the test 4

6)Testing allNeededCoord
a)Length of allNeededCoord should be the same as length of SeedPoints,
b) size of each subentry given z=1 should be the same as in test 4 
c)Given some point in sub sub entry is (1,1,1) should pass the test 4

7)Testing calculatePatchStatistics(allNeededCoord,I)
We supply 3 dimensional matrix of ones  as I and supply all of the coordinates we should get back the mean =1 and std = 0

8)We supply set of different coordinates to 3 dimensional array of ones we should get back two dimensional list where each sublist should have first entry 1 and second 0

9)calculateFeatureVectors(patchStats) - if we will keep all matrices as matrices of ones after reduction and norm we still should have get mean of 1 and std of 0

10) For vector 1,2,0 mean vector should be 0,1,-1

11)If we will supply vectors of ones the covariance matrix should have all entries =0

12)We will supply as a feature vector vector filled with randomly generated numbers of given dimensionality on the basis of which we will calculate normalizing constant - next in a loop we will randomly crate x vector of appropriate dimensionality and we will evaluate gaussian pdf with calculated normalizing constant test will be passed if calculated values will be always between 0 and 1

13)We apply tests from 12 to each entry

14)We supply to our log normal distributions some mean and covariance and we evaluate pdf on some entries - and compare to the results of log multivariate gaussian from Distributions.jl package
```


```@doc
2)Example 3 dimensional matrix $I_i$ with example marking mask $M_i$ will be saved with known number of marked points  x on known position
We will assert thag wefound correctly those points and that 
```
@testset " getCoordinatesOfMarkings " begin 
A = ones(4,4,4)
B = ones(4,4,4)
coords = [CartesianIndex(1,2,3),CartesianIndex(1,4,3) ] 
A[coords].=7
@test Main.GaussianPure()
end # getCoordinatesOfMarkings


```@doc
For coord in arr 
	Assert equals z value of coord = zSlice
end for 
    ```

    ```@doc
3)We will test getOneNormDist(pointA,pointB) by manually supplying some points with know 1 norm distance for example
Assert equal getOneNormDist(CartesianCoordinate(1,1,1),CartesianCoordinate(2,2,2)) == 3
```

```@doc
4)Testing getCartesianAroundPoint(point,z) given coordinate (1,1,1) and distance 1 we should get in result set containing { (1,1,1),(0,1,1),(2,1,1),(1,0,1),(1,2,1),(1,1,0),(1,1,2) }
```

```@doc
5)Testing primPatch 
a)Length of primPatch  should be the same as length of SeedPoints,
b) size of each entry given z=1 should be the same as in test 4 
c)Given some point in SeedPoints is (1,1,1) should pass the test 4
```

```@doc
6)Testing allNeededCoord
a)Length of allNeededCoord should be the same as length of SeedPoints,
b) size of each subentry given z=1 should be the same as in test 4 
c)Given some point in sub sub entry is (1,1,1) should pass the test 4
```

```@doc
7)Testing calculatePatchStatistics(allNeededCoord,I)
We supply 3 dimensional matrix of ones  as I and supply all of the coordinates we should get back the mean =1 and std = 0
```

```@doc
8)We supply set of different coordinates to 3 dimensional array of ones we should get back two dimensional list where each sublist should have first entry 1 and second 0
```

```@doc
9)calculateFeatureVectors(patchStats) - if we will keep all matrices as matrices of ones after reduction and norm we still should have get mean of 1 and std of 0
```

```@doc
10) For vector 1,2,0 mean vector should be 0,1,-1
```

```@doc
11)If we will supply vectors of ones the covariance matrix should have all entries =0
```

```@doc
12)We will supply as a feature vector vector filled with randomly generated numbers of given dimensionality on the basis of which we will calculate normalizing constant - next in a loop we will randomly crate x vector of appropriate dimensionality and we will evaluate gaussian pdf with calculated normalizing constant test will be passed if calculated values will be always between 0 and 1
```

```@doc
13)We apply tests from 12 to each entry
```

```@doc
14)We supply to our log normal distributions some mean and covariance and we evaluate pdf on some entries - and compare to the results of log multivariate gaussian from Distributions.jl package
```
