

```@doc
module that will apply earlier calculated gaussian multivariate distribution to means and standard deviations  of patches in the image
  ```
module applyGaussian 
```@doc
Remember to ignore all of the patches that has exactly equal value of whole patch

  ```
using DrWatson
@quickactivate "Probabilistic medical segmentation"
using CUDA
using ParallelStencil
using ParallelStencil.FiniteDifferences3D
using Main.generalUtils
using KernelAbstractions
using MacroTools

```@doc
Configuring the ParallelStencil library 
  ```
USE_GPU = true  # Use GPU? If this is set false, then no GPU needs to be available

function configureParSten()  
  @static if USE_GPU
      @init_parallel_stencil(CUDA, Float64, 3)
  else
      @init_parallel_stencil(Threads, Float64, 3)
  end
end

configureParSten() 


#cartList = cartesianCoordAroundPoint(CartesianIndex(0,0,0),1)

```@doc
main function that works on the predefined patch - calculates means, standard deviations in order to evaluate gaussian pdf's and choose biggest 
as inputs we will also get list of arguments we will get 4 arrays , the length of this array will be arbitrary set to 3 - so we will evaluate only 3 points in the organ
in each list we will sotre 
  1. mean vector \( feature vector minus its mean\)
  2. covariance matrix inverse
  3. log of normalization constant
  4.covariance matrix

  
  ```
  @parallel_indices (ix,iy,iz) function computeMeanStd!(In::Data.Array,indArr)
      # 4-point Neuman stencil
      if (ix<=size(Mean,1) && iy<=size(Mean,2) && iz<=size(Mean,3))
          ixi, iyi, izi = ix+1, iy+1, iz+1

      

            Mean[ix,iy,iz] = (In[ixi-1,iyi  ,izi  ] +
                              In[ixi-1,iyi  ,izi  ] + In[ixi+1,iyi  ,izi  ] + In[ixi  ,iyi-1,izi  ] + In[ixi  ,iyi+1,izi  ] +
                              In[ixi  ,iyi  ,izi-1 ] + In[ixi  ,iyi  ,izi+1])/7.0
    
            Std[ix,iy,iz]  = ((In[ixi-1,iyi  ,izi  ] - Mean[ix,iy,iz])^2 +
                              (In[ixi-1,iyi  ,izi  ] - Mean[ix,iy,iz])^2 +
                              (In[ixi+1,iyi  ,izi  ] - Mean[ix,iy,iz])^2 +
                              (In[ixi  ,iyi-1,izi  ] - Mean[ix,iy,iz])^2 + 
                              (In[ixi  ,iyi+1,izi  ] - Mean[ix,iy,iz])^2 +
                              (In[ixi  ,iyi  ,izi-1] - Mean[ix,iy,iz])^2 + 
                              (In[ixi  ,iyi  ,izi+1] - Mean[ix,iy,iz])^2)/7.0

      end
      return
  end







  @time   img_process()
#manuallySet    0.000949 seconds (959 allocations: 38.281 KiB)




       # Numerics
  nx, ny, nz = 64, 64, 64
       # Array allocations
  In    =  @rand(nx  ,ny  ,nz  )
  MeanC  = @zeros(nx-2,ny-2,nz-2)
  Std   = @zeros(nx-2,ny-2,nz-2)


#we get catrtesian indicies around 0,0,0 cartesian index  
indArr=   cartesianCoordAroundPoint(CartesianIndex(0,0,0),1) |>
  (cartIndicises)-> map((ind)->[ind[1],ind[2],ind[3]],cartIndicises) |>
  (arr) ->CuArray(vecvec_to_matrix(arr))

  ix, iy, iz = 1, 1, 1


  KernelAbstractions.Extras.LoopInfo.@unroll for ind in indArr

    print(ind)

  end  

@views function img_process()
  @parallel computeMeanStd!(MeanC, Std, In,indArr)
  return
end

MeanB==MeanC


  @time img_process()










  ```@doc
  point - cartesian coordinates of point around which we want the cartesian coordeinates
  return set of cartetian coordinates of given distance -patchSize from a point
```
function cartesianCoordAroundPoint(pointCart::CartesianIndex{3}, patchSize ::Int)::Array{CartesianIndex{3}}
  ones = CartesianIndex(patchSize,patchSize,patchSize) # cartesian 3 dimensional index used for calculations to get range of the cartesian indicis to analyze
  out = Array{CartesianIndex{3}}(UndefInitializer(), 6+2*patchSize^4)
  index =0
  for J in (pointCart-ones):(pointCart+ones)
    diff = J - pointCart # diffrence between dimensions relative to point of origin
      if cartesianTolinear(diff) <= patchSize
        index+=1
        out[index] = J
      end
      end
return out[1:index]
end

```@doc
works only for 3d cartesian coordinates
  cart - cartesian coordinates of point where we will add the dimensions ...
```
function cartesianTolinear(pointCart::CartesianIndex{3}) :: Int16
   abs(pointCart[1])+ abs(pointCart[2])+abs(pointCart[3])
end



end #applyGaussian