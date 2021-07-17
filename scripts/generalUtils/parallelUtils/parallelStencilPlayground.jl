using ParallelStencil
using MacroTools
"""

ParallelKernel
basic functions

shared 
syntactic sugar

function files

init_parallel_stencil
parallel
reset_parallel_stencil


FiniteDifferences
include for example indicies


"""


"""
in parallel ParallelKernel
first Data and Exceptions

than shared


than 
allocators
hide_communication - for example here boundary checking is done
init_parallel_kernel- for example setting proper data type for device
parallel
reset_parallel_kernel


"""



"""
Data struct - is just definition of the array that is independent of whether parallel CPU or GPU


"""
const RAND_DOC = """
So here we see how with macro we can specialize the function to CUDA or CPU ...


"""

@doc RAND_DOC
macro rand(args...) check_initialized(); esc(_rand(args...)); end

macro rand_cuda(args...)     check_initialized(); esc(_rand(args...; package=PKG_CUDA)); end
macro rand_threads(args...)  check_initialized(); esc(_rand(args...; package=PKG_THREADS)); end
"""and as a function  """

function _rand(args...; package::Symbol=get_package())
    numbertype = get_numbertype()
    if     (package == PKG_CUDA)    return :(CUDA.CuArray(rand($numbertype, $(args...))))
    elseif (package == PKG_THREADS) return :(Base.rand($numbertype, $(args...)))
    else                            @KeywordArgumentError("$ERRMSG_UNSUPPORTED_PACKAGE (obtained: $package).")
    end
end

"""
from hihecommunication

- `boundary_width::Tuple{Integer,Integer,Integer} | Tuple{Integer,Integer} | Tuple{Integer}`: width of the boundaries in each dimension. The boundaries must include (at least) all the data that is accessed in the communcation performed.
- `block`: code block wich starts with exactly one [`@parallel`](@ref) call to perform computations, followed by code to set boundary conditions and to perform communication (as e.g. `update_halo!` from the package `ImplicitGlobalGrid`). The [`@parallel`](@ref) call to perform computations cannot contain any positional arguments (ranges, nblocks or nthreads) nor the stream keyword argument (stream=...). The code to set boundary conditions and to perform communication must only access the elements in the boundary ranges of the fields modified in the [`@parallel`](@ref) call; all elements can be acccessed from other fields. Moreover, this code must not include statements in array broadcasting notation, because they are always run on the default CUDA stream (for CUDA.jl < v2.0), which makes CUDA stream overlapping impossible. Instead, boundary region elements can, e.g., be accessed with [`@parallel`](@ref) calls passing a ranges argument that ensures that no threads mapping to elements outside of `ranges_outer` are launched. Note that these [`@parallel`](@ref) `ranges` calls cannot contain any other positional arguments (nblocks or nthreads) nor the stream keyword argument (stream=...).


- `ranges_outer::`Tuple with one or multiple `ranges` as required by the corresponding argument of [`@parallel`](@ref): the `ranges` must together span (at least) all the data that is accessed in the communcation and boundary conditions performed.
- `ranges_inner::`Tuple with one or multiple `ranges` as required by the corresponding argument of [`@parallel`](@ref): the `ranges` must together span the data that is not included by `ranges_outer`.

"""


"""
from parallel

!!! note "Advanced optional arguments"
    - `ranges::Tuple{UnitRange{},UnitRange{},UnitRange{}} | Tuple{UnitRange{},UnitRange{}} | Tuple{UnitRange{}} | UnitRange{}`: the ranges of indices in each dimension for which computations must be performed.
    - `nblocks::Tuple{Integer,Integer,Integer}`: the number of blocks to be used if the package CUDA was selected with [`@init_parallel_kernel`](@ref).
    - `nthreads::Tuple{Integer,Integer,Integer}`: the number of threads to be used if the package CUDA was selected with [`@init_parallel_kernel`](@ref).
    - `kwargs...`: keyword arguments to be passed further to CUDA (ignored for Threads).

"""


"""
some experiments with computing ranges

"""
ParallelStencil.ParallelKernel.compute_ranges(6) #(1:6, 1:1, 1:1)
ParallelStencil.ParallelKernel.compute_ranges((40,40,6)) #(1:40, 1:40, 1:6)



"""
from FiniteDifferences2D

inner elements seesm to be all apart from borders

"""

@doc "`@d_xa(A)`: Compute differences between adjacent elements of `A` along the dimension x." :(@d_xa)
@doc "`@d_ya(A)`: Compute differences between adjacent elements of `A` along the dimension y." :(@d_ya)
@doc "`@d_xi(A)`: Compute differences between adjacent elements of `A` along the dimension x and select the inner elements of `A` in the remaining dimension. Corresponds to `@inn_y(@d_xa(A))`." :(@d_xi)
@doc "`@d_yi(A)`: Compute differences between adjacent elements of `A` along the dimension y and select the inner elements of `A` in the remaining dimension. Corresponds to `@inn_x(@d_ya(A))`." :(@d_yi)
@doc "`@d2_xi(A)`: Compute the 2nd order differences between adjacent elements of `A` along the dimension x and select the inner elements of `A` in the remaining dimension. Corresponds to `@inn_y(@d2_xa(A))`." :(@d2_xi)
@doc "`@d2_yi(A)`: Compute the 2nd order differences between adjacent elements of `A` along the dimension y and select the inner elements of `A` in the remaining dimension. Corresponds to `@inn_x(@d2_ya(A))`." :(@d2_yi)
@doc "`@all(A)`: Select all elements of `A`. Corresponds to `A[:,:]`." :(@all)
@doc "`@inn(A)`: Select the inner elements of `A`. Corresponds to `A[2:end-1,2:end-1]`." :(@inn)
@doc "`@inn_x(A)`: Select the inner elements of `A` in dimension x. Corresponds to `A[2:end-1,:]`." :(@inn_x)
@doc "`@inn_y(A)`: Select the inner elements of `A` in dimension y. Corresponds to `A[:,2:end-1]`." :(@inn_y)
@doc "`@av(A)`: Compute averages between adjacent elements of `A` along the dimensions x and y." :(@av)
@doc "`@av_xa(A)`: Compute averages between adjacent elements of `A` along the dimension x." :(@av_xa)
@doc "`@av_ya(A)`: Compute averages between adjacent elements of `A` along the dimension y." :(@av_ya)
@doc "`@av_xi(A)`: Compute averages between adjacent elements of `A` along the dimension x and select the inner elements of `A` in the remaining dimension. Corresponds to `@inn_y(@av_xa(A))`." :(@av_xi)
@doc "`@av_yi(A)`: Compute averages between adjacent elements of `A` along the dimension y and select the inner elements of `A` in the remaining dimension. Corresponds to `@inn_x(@av_ya(A))`." :(@av_yi)
@doc "`@maxloc(A)`: Compute the maximum between 2nd order adjacent elements of `A`, using a moving window of size 3." :(@maxloc)
@doc "`@minloc(A)`: Compute the minimum between 2nd order adjacent elements of `A`, using a moving window of size 3." :(@minloc)


"""
FiniteDifferences3D
"""
@doc "`@d_xa(A)`: Compute differences between adjacent elements of `A` along the dimension x." :(@d_xa)
@doc "`@d_ya(A)`: Compute differences between adjacent elements of `A` along the dimension y." :(@d_ya)
@doc "`@d_za(A)`: Compute differences between adjacent elements of `A` along the dimension z." :(@d_za)
@doc "`@d_xi(A)`: Compute differences between adjacent elements of `A` along the dimension x and select the inner elements of `A` in the remaining dimensions. Corresponds to `@inn_yz(@d_xa(A))`." :(@d_xi)
@doc "`@d_yi(A)`: Compute differences between adjacent elements of `A` along the dimension y and select the inner elements of `A` in the remaining dimensions. Corresponds to `@inn_xz(@d_ya(A))`." :(@d_yi)
@doc "`@d_zi(A)`: Compute differences between adjacent elements of `A` along the dimension z and select the inner elements of `A` in the remaining dimensions. Corresponds to `@inn_xy(@d_za(A))`." :(@d_zi)
@doc "`@d2_xi(A)`: Compute the 2nd order differences between adjacent elements of `A` along the dimension x and select the inner elements of `A` in the remaining dimensions. Corresponds to `@inn_yz(@d2_xa(A))`." :(@d2_xi)
@doc "`@d2_yi(A)`: Compute the 2nd order differences between adjacent elements of `A` along the dimension y and select the inner elements of `A` in the remaining dimensions. Corresponds to `@inn_xz(@d2_ya(A))`." :(@d2_yi)
@doc "`@d2_zi(A)`: Compute the 2nd order differences between adjacent elements of `A` along the dimension y and select the inner elements of `A` in the remaining dimensions. Corresponds to `@inn_xy(@d2_za(A))`." :(@d2_zi)
@doc "`@all(A)`: Select all elements of `A`. Corresponds to `A[:,:,:]`." :(@all)
@doc "`@inn(A)`: Select the inner elements of `A`. Corresponds to `A[2:end-1,2:end-1,2:end-1]`." :(@inn)
@doc "`@inn_x(A)`: Select the inner elements of `A` in dimension x. Corresponds to `A[2:end-1,:,:]`." :(@inn_x)
@doc "`@inn_y(A)`: Select the inner elements of `A` in dimension y. Corresponds to `A[:,2:end-1,:]`." :(@inn_y)
@doc "`@inn_z(A)`: Select the inner elements of `A` in dimension z. Corresponds to `A[:,:,2:end-1]`." :(@inn_z)
@doc "`@inn_xy(A)`: Select the inner elements of `A` in dimensions x and y. Corresponds to `A[2:end-1,2:end-1,:]`." :(@inn_xy)
@doc "`@inn_xz(A)`: Select the inner elements of `A` in dimensions x and z. Corresponds to `A[2:end-1,:,2:end-1]`." :(@inn_xz)
@doc "`@inn_yz(A)`: Select the inner elements of `A` in dimensions y and z. Corresponds to `A[:,2:end-1,2:end-1]`." :(@inn_yz)
@doc "`@av(A)`: Compute averages between adjacent elements of `A` along the dimensions x and y and z." :(@av)
@doc "`@av_xa(A)`: Compute averages between adjacent elements of `A` along the dimension x." :(@av_xa)
@doc "`@av_ya(A)`: Compute averages between adjacent elements of `A` along the dimension y." :(@av_ya)
@doc "`@av_za(A)`: Compute averages between adjacent elements of `A` along the dimension z." :(@av_za)
@doc "`@av_xi(A)`: Compute averages between adjacent elements of `A` along the dimension x and select the inner elements of `A` in the remaining dimensions. Corresponds to `@inn_yz(@av_xa(A))`." :(@av_xi)
@doc "`@av_yi(A)`: Compute averages between adjacent elements of `A` along the dimension y and select the inner elements of `A` in the remaining dimensions. Corresponds to `@inn_xz(@av_ya(A))`." :(@av_yi)
@doc "`@av_zi(A)`: Compute averages between adjacent elements of `A` along the dimension z and select the inner elements of `A` in the remaining dimensions. Corresponds to `@inn_xy(@av_za(A))`." :(@av_zi)
@doc "`@av_xya(A)`: Compute averages between adjacent elements of `A` along the dimensions x and y." :(@av_xya)
@doc "`@av_xza(A)`: Compute averages between adjacent elements of `A` along the dimensions x and z." :(@av_xza)
@doc "`@av_yza(A)`: Compute averages between adjacent elements of `A` along the dimensions y and z." :(@av_yza)
@doc "`@av_xyi(A)`: Compute averages between adjacent elements of `A` along the dimensions x and y and select the inner elements of `A` in the remaining dimension. Corresponds to `@inn_z(@av_xya(A))`." :(@av_xyi)
@doc "`@av_xzi(A)`: Compute averages between adjacent elements of `A` along the dimensions x and z and select the inner elements of `A` in the remaining dimension. Corresponds to `@inn_y(@av_xza(A))`." :(@av_xzi)
@doc "`@av_yzi(A)`: Compute averages between adjacent elements of `A` along the dimensions y and z and select the inner elements of `A` in the remaining dimension. Corresponds to `@inn_x(@av_yza(A))`." :(@av_yzi)
@doc "`@maxloc(A)`: Compute the maximum between 2nd order adjacent elements of `A`, using a moving window of size 3." :(@maxloc)
@doc "`@minloc(A)`: Compute the minimum between 2nd order adjacent elements of `A`, using a moving window of size 3." :(@minloc)




const USE_GPU = false
using ImplicitGlobalGrid, Plots
import MPI
using ParallelStencil
using ParallelStencil.FiniteDifferences3D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 3);
else
    @init_parallel_stencil(Threads, Float64, 3);
end

nx,ny,nz = 3,3,3

T   = @ones(nx, ny, nz);
T2   = @zeros(nx, ny, nz);

#Basically If I get it 
@parallel function diffusion3D(T2, T)
    @inn(T2) = @av(T);
    return
end

@parallel diffusion3D(T2, T)





@parallel_indices (ix,iy,iz) function computeMeanStd!(Mean::Data.Array, Std::Data.Array, In::Data.Array)
    # 7-point Neuman stencil
    if (ix<=size(Mean,1) && iy<=size(Mean,2) && iz<=size(Mean,3))
        ixi, iyi, izi = ix+1, iy+1, iz+1
        Mean[ix,iy,iz] = (In[ixi,iyi  ,izi  ] + #
                          In[ixi-1,iyi  ,izi  ] + #
                          In[ixi+1,iyi  ,izi  ] +
                          In[ixi  ,iyi-1,izi  ] + 
                          In[ixi  ,iyi+1,izi  ] +
                          In[ixi  ,iyi  ,izi-1] + 
                          In[ixi  ,iyi  ,izi+1])/7.0

        Std[ix,iy,iz]  = ((In[ixi-1,iyi  ,izi  ] - Mean[ix,iy,iz])^2 +
                          (In[ixi-1,iyi  ,izi  ] - Mean[ix,iy,iz])^2 + (In[ixi+1,iyi  ,izi  ] - Mean[ix,iy,iz])^2 +
                          (In[ixi  ,iyi-1,izi  ] - Mean[ix,iy,iz])^2 + (In[ixi  ,iyi+1,izi  ] - Mean[ix,iy,iz])^2 +
                          (In[ixi  ,iyi  ,izi-1] - Mean[ix,iy,iz])^2 + (In[ixi  ,iyi  ,izi+1] - Mean[ix,iy,iz])^2)/7.0
    end
    return
end

macro myAv(A::Symbol,ix::Symbol,iy::Symbol,iz::Symbol    )  
    esc(:((
        $A[$ix  ,$iy  ,$iz  ] +
        $A[$ix-1  ,$iy  ,$iz  ] +
        $A[$ix+1  ,$iy  ,$iz  ] +
        $A[$ix  ,$iy-1  ,$iz  ] +
        $A[$ix  ,$iy+1  ,$iz  ] +
        $A[$ix  ,$iy  ,$iz-1  ] +
        $A[$ix  ,$iy  ,$iz+1  ] 
    )/7.0)) end


nx, ny, nz = 64, 64, 64
# Array allocations
In    =  @ones(nx  ,ny  ,nz  )
MeanB  = @zeros(nx-2,ny-2,nz-2)
StdB   = @zeros(nx-2,ny-2,nz-2)
MeanC  = @zeros(nx-2,ny-2,nz-2)
StdC   = @zeros(nx-2,ny-2,nz-2)

# Calculation
@parallel MycomputeMeanStd!(MeanB, StdB, In)
@parallel computeMeanStd!(MeanC, StdC, In)

MeanC
MeanB

MeanC==MeanB
StdC == StdB


 @parallel_indices (ix,iy,iz) function MycomputeMeanStd!(Mean::Data.Array, Std::Data.Array, In::Data.Array)
    # 7-point Neuman stencil
    if (ix<=size(Mean,1) && iy<=size(Mean,2) && iz<=size(Mean,3))
        ixi, iyi, izi = ix+1, iy+1, iz+1

        Mean[ix,iy,iz] = @myAv(In,ixi,iyi,izi)

        Std[ix,iy,iz]  = ((In[ixi-1,iyi  ,izi  ] - Mean[ix,iy,iz])^2 +
                          (In[ixi-1,iyi  ,izi  ] - Mean[ix,iy,iz])^2 + (In[ixi+1,iyi  ,izi  ] - Mean[ix,iy,iz])^2 +
                          (In[ixi  ,iyi-1,izi  ] - Mean[ix,iy,iz])^2 + (In[ixi  ,iyi+1,izi  ] - Mean[ix,iy,iz])^2 +
                          (In[ixi  ,iyi  ,izi-1] - Mean[ix,iy,iz])^2 + (In[ixi  ,iyi  ,izi+1] - Mean[ix,iy,iz])^2)/7.0
    end
    return
end