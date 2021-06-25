using DrWatson
@quickactivate "Probabilistic medical segmentation"
dirToImageHelper = DrWatson.scriptsdir("display","imageViewerHelper.jl")
include(dirToImageHelper)
using Main.imageViewerHelper
using Test
using  Observables

@testset "imageViewerHelper"  begin



    @testset "imageViewerHelper"  begin
        @test Main.imageViewerHelper.cartesianTolinear(CartesianIndex(1,1,1)) ==3
        @test Main.imageViewerHelper.cartesianTolinear(CartesianIndex(-1,1,2)) ==4
    end

    @testset "imageViewerHelper"  begin
        @test Set(Main.imageViewerHelper.cartesianCoordAroundPoint(CartesianIndex(0,0,0),1)) == Set([CartesianIndex(0,0,0),CartesianIndex(0,0,1),
         CartesianIndex(0,0,-1),CartesianIndex(0,1,0),
         CartesianIndex(0,-1,0),CartesianIndex(1,0,0),
         CartesianIndex(-1,0,0)])
    end
    
    @testset "markMaskArrayPatchTo!"  begin
        dims = size(ones(3,3,3))
        points1 = [CartesianIndex(1,1,1)]
        pointsMulti = [CartesianIndex(1,1,1),CartesianIndex(2,1,2)]
        pointsTwoGoodOneOut = [CartesianIndex(1,1,1),CartesianIndex(3,3,3),CartesianIndex(0,1,2),CartesianIndex(5,1,2)]# point that is outside of the scope
     
        valueToSet = 5

        Arr1 = ones(3,3,3)
        Arr1[1,1,1] = 5
        @test Main.imageViewerHelper.markMaskArrayPatchTo!(ones(3,3,3), points1, valueToSet,dims)  ==  Arr1
        Arr2 = ones(3,3,3)
        Arr2[ [CartesianIndex(1,1,1),CartesianIndex(2,1,2)] ].=5
        @test Main.imageViewerHelper.markMaskArrayPatchTo!(ones(3,3,3), pointsMulti, valueToSet,dims)  ==  Arr2
        Arr3 = ones(3,3,3)
        Arr3[[ CartesianIndex(1,1,1),CartesianIndex(3,3,3) ]].=5
        @test Main.imageViewerHelper.markMaskArrayPatchTo!(ones(3,3,3), pointsTwoGoodOneOut, valueToSet,dims)  ==  Arr3
    end

    @testset "calculateMouseAndSetmask"  begin
        Arr = Observable(ones(5,5,5))
        dims = size(Arr[])
        sliceNumb= Observable(3)
        compBoxWidth = 510 
        compBoxHeight = 510 
        pixelsNumbInX =dims[1]
        pixelsNumbInY =dims[2]
        xMouse = (3*compBoxWidth)/pixelsNumbInX
        yMouse = (3*compBoxHeight)/pixelsNumbInY

        modified = Main.imageViewerHelper.calculateMouseAndSetmask(Arr,dims,sliceNumb, xMouse,yMouse,1,compBoxWidth, compBoxHeight)  
        @test modified[3,3,3] == 6 
        @test modified[3,4,3] == 5 
        @test modified[4,3,4] == 4 
        @test modified[3,5,3] == 4 
        @test modified[3,3,2] == 5 
    end



end