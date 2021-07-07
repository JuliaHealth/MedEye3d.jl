
using DrWatson
@quickactivate "Probabilistic medical segmentation"
using Distributions
using CUDA
using NNlib
using GPUArrays
using BenchmarkTools
```@doc
Remember to ignore all of the patches that has exactly equal value of whole patch

  ```


  ```@doc

  
adapted from https://discourse.julialang.org/t/3d-medical-app-stencil/64019
    ```

    const USE_GPU = true  # Use GPU? If this is set false, then no GPU needs to be available
    using ParallelStencil
    using ParallelStencil.FiniteDifferences3D
    @static if USE_GPU
        @init_parallel_stencil(CUDA, Float64, 3)
    else
        @init_parallel_stencil(Threads, Float64, 3)
    end
   
    @parallel_indices (ix,iy,iz) function computeMeanStd!(Mean::Data.Array, Std::Data.Array, In::Data.Array)
        # 4-point Neuman stencil
        if (ix<=size(Mean,1) && iy<=size(Mean,2) && iz<=size(Mean,3))
            ixi, iyi, izi = ix+1, iy+1, iz+1
            Mean[ix,iy,iz] = (In[ixi-1,iyi  ,izi  ] +
                              In[ixi-1,iyi  ,izi  ] + In[ixi+1,iyi  ,izi  ] +
                              In[ixi  ,iyi-1,izi  ] + In[ixi  ,iyi+1,izi  ] +
                              In[ixi  ,iyi  ,izi-1] + In[ixi  ,iyi  ,izi+1])/7.0
    
            Std[ix,iy,iz]  = ((In[ixi-1,iyi  ,izi  ] - Mean[ix,iy,iz])^2 +
                              (In[ixi-1,iyi  ,izi  ] - Mean[ix,iy,iz])^2 + (In[ixi+1,iyi  ,izi  ] - Mean[ix,iy,iz])^2 +
                              (In[ixi  ,iyi-1,izi  ] - Mean[ix,iy,iz])^2 + (In[ixi  ,iyi+1,izi  ] - Mean[ix,iy,iz])^2 +
                              (In[ixi  ,iyi  ,izi-1] - Mean[ix,iy,iz])^2 + (In[ixi  ,iyi  ,izi+1] - Mean[ix,iy,iz])^2)/7.0
        end
        return
    end
    
    @views function img_process()
        # Numerics
        nx, ny, nz = 64, 64, 64
        # Array allocations
        In    =  @rand(nx  ,ny  ,nz  )
        Mean  = @zeros(nx-2,ny-2,nz-2)
        Std   = @zeros(nx-2,ny-2,nz-2)
        # Calculation
        @parallel computeMeanStd!(Mean, Std, In)
        # Visualisation
        p1 = heatmap(1:nx  , 1:nz  , Array(In )[:,Int(round(ny/2)),:]'    , aspect_ratio=1, xlims=(1,nx)  , ylims=(1,nz)  , c=:viridis, title="Input data")
        p2 = heatmap(1:nx-2, 1:nz-2, Array(Std)[:,Int(round((ny-2)/2)),:]', aspect_ratio=1, xlims=(1,nx-2), ylims=(1,nz-2), c=:viridis, title="Standard deviation")
        display(plot(p1, p2))
        return
    end
    
    @time img_process()