
using DrWatson
@quickactivate "Probabilistic medical segmentation"
using Distributions
using CUDA
using NNlib
using GPUArrays
using BenchmarkTools
using LinearAlgebra
using LinearAlgebra: BlasInt

```@doc
Remember to ignore all of the patches that has exactly equal value of whole patch

  ```


  ```@doc

  
adapted from https://discourse.julialang.org/t/3d-medical-app-stencil/64019
    ```

    const USE_GPU = true  # Use GPU? If this is set false, then no GPU needs to be available
    using ParallelStencil
    using ParallelStencil.FiniteDifferences3D
    using MacroTools

    @static if USE_GPU
        @init_parallel_stencil(CUDA, Float64, 3)
    else
        @init_parallel_stencil(Threads, Float64, 3)
    end
    cartList = cartesianCoordAroundPoint(CartesianIndex(0,0,0),1)

    @parallel_indices (ix,iy,iz) function computeMeanStd!(Mean::Data.Array, Std::Data.Array, In::Data.Array)
        # 4-point Neuman stencil
        if (ix<=size(Mean,1) && iy<=size(Mean,2) && iz<=size(Mean,3))
            ixi, iyi, izi = ix+1, iy+1, iz+1
            #aa = aa*bb
            a = 1
            @indexIng(a)

end
return
end

@time img_process()


macro add(inn)
    return :( $inn +1 )
end



   # eval(Meta.parse(:($a + b)))0+*
            
            # Mean[ix,iy,iz] = (In[CartesianIndex(ixi-1,iyi  ,izi ) ] +
            #                   In[CartesianIndex(ixi-1,iyi  ,izi ) ] + In[ixi+1,iyi  ,izi  ] + In[CartesianIndex(ixi  ,iyi-1,izi ) ] + In[ixi  ,iyi+1,izi  ] +
            #                   In[CartesianIndex(ixi  ,iyi  ,izi-1 )] + In[ixi  ,iyi  ,izi+1])/7.0
    
            # Std[ix,iy,iz]  = ((In[ixi-1,iyi  ,izi  ] - Mean[ix,iy,iz])^2 +
            #                   (In[ixi-1,iyi  ,izi  ] - Mean[ix,iy,iz])^2 + (In[ixi+1,iyi  ,izi  ] - Mean[ix,iy,iz])^2 +
            #                   (In[ixi  ,iyi-1,izi  ] - Mean[ix,iy,iz])^2 + (In[ixi  ,iyi+1,izi  ] - Mean[ix,iy,iz])^2 +
            #                   (In[ixi  ,iyi  ,izi-1] - Mean[ix,iy,iz])^2 + (In[ixi  ,iyi  ,izi+1] - Mean[ix,iy,iz])^2)/7.0




         # Numerics
    nx, ny, nz = 64, 64, 64
         # Array allocations
    In    =  @rand(nx  ,ny  ,nz  )
    MeanB  = @zeros(nx-2,ny-2,nz-2)
    Std   = @zeros(nx-2,ny-2,nz-2)
    x_d = CUDA.rand(2)
    x_y = CUDA.rand(2)

    @views function img_process()
   

        # Calculation
        @parallel computeMeanStd!(MeanB, Std, In)

        return
    end
    
maximum(Mean)
maximum(In)


    @time img_process()


    #0.356854 seconds (322.37 k allocations: 16.921 MiB, 10.07% compilation time)
    MeanA = Mean #    0.000930 seconds (745 allocations: 18.922 KiB)

    MeanA == MeanB



    
    
    
    cartList+CartesianIndex(2,2,2)




    ex2 = Expr(:call, :+, 1, 1)
    dump(ex2)
a = 1
b= 1
c= 1

    eval(Meta.parse("a + b*c + 1"))