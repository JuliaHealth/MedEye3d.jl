
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
according to https://github.com/omlins/ParallelStencil.jl

All the miniapps can be interactively executed within the Julia REPL (this includes the multi-XPU versions when using a single CPU or GPU). Note that for optimal performance the miniapp script of interest <miniapp_code> should be launched from the shell using the project's dependencies --project, disabling array bound checking --check-bounds=no, and using optimization level 3 -O3.

 julia --project --check-bound=no -O3 <miniapp_code>.jl

  

    ```

    const USE_GPU = true
    using ParallelStencil
    @static if USE_GPU
        @init_parallel_stencil(CUDA, Float64, 3);
    else
        @init_parallel_stencil(Threads, Float64, 3);
    end
    
    @parallel_indices (ix,iy,iz) function copy3D!(T2, T, Ci)
        T2[ix,iy,iz] = T[ix,iy,iz] + Ci[ix,iy,iz];
        return
    end
    
    function memcopy3D()
    # Numerics
    nx, ny, nz = 212, 212, 212;                              # Number of gridpoints in dimensions x, y and z
    nt  = 100;                                               # Number of time steps
    
    # Array initializations
    T   = @zeros(nx, ny, nz);
    T2  = @zeros(nx, ny, nz);
    Ci  = @zeros(nx, ny, nz);
    
    # Initial conditions
    Ci .= 1/2.0;
    T  .= 1.7;
    T2 .= T;
    
    # Time loop
    for it = 1:nt
        if (it == 11) global t0=time(); end  # Start measuring time.
        @parallel copy3D!(T2, T, Ci);
        T, T2 = T2, T;
    end
    time_s=time()-t0
    
    # Performance
    A_eff = (2*1+1)*1/1e9*nx*ny*nz*sizeof(Data.Number);      # Effective main memory access per iteration [GB] (Lower bound of required memory access: T has to be read and written: 2 whole-array memaccess; Ci has to be read: : 1 whole-array memaccess)
    t_it  = time_s/(nt-10);                                  # Execution time per iteration [s]
    T_eff = A_eff/t_it;                                      # Effective memory throughput [GB/s]
    println("time_s=$time_s T_eff=$T_eff");
    end
    
    memcopy3D()





    
    nx, ny, nz = 4, 4, 4;                              # Number of gridpoints in dimensions x, y and z
  
    # Array initializations

    @parallel_indices (iy,iz) function bc_x!()
    print(A[1  , iy, iz])
    # A[end, iy, iz] = 3
    return
    end
    bc_x!(T)