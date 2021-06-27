
using DrWatson
@quickactivate "Probabilistic medical segmentation"
using Distributions
using CUDA
using NNlib
using GPUArrays

```@doc
Remember to ignore all of the patches that has exactly equal value of whole patch

  ```


N = 2^20
x_d = CUDA.fill(1.0f0, N)  # a vector stored on the GPU filled with 1.0 (Float32)
y_d = CUDA.fill(2.0f0, N)  # a vector stored on the GPU filled with 2.0



function gpu_add1!(y, x)
  for i = 1:length(y)
      @inbounds y[i] += x[i]
  end
  return nothing
end

fill!(y_d, 2)
@cuda gpu_add1!(y_d, x_d)
using Test
@test all(Array(y_d) .== 3.0f0)

#benchmarking
function bench_gpu1!(y, x)
  CUDA.@sync begin
      @cuda gpu_add1!(y, x)
  end
end



using Latexify
print(latexify("x+y/(b-2)^2"))



using NNlib: DenseConvDims
using Flux 
# observation one is that the second argument can not be bigger than a first 
a, b ,c = ones(Float64, 4,2,1), ones(Float64, 2, 2, 1), ones(Float64, 2, 2, 1)
cdims = DenseConvDims(a, b)
z = NNlib.depthwiseconv(a, b,c, cdims)
cdims
z[1,:,:]


da, db = CuArray(a), CuArray(b)

collect(NNlib.conv(da, db, cdims))

# Return the convolution of filter `w` with tensor `x`, overwriting `y` if provided, according
# to keyword arguments or the convolution descriptor `d`. Optionally perform bias addition,
# activation and/or scaling:

# All tensors should have the same number of dimensions. If they are less than 4-D their
# dimensions are assumed to be padded on the left with ones. `x` has size `(X...,Cx,N)` where
# `(X...)` are the spatial dimensions, `Cx` is the number of input channels, and `N` is the
# number of instances. `y,z` have size `(Y...,Cy,N)` where `(Y...)` are the spatial dimensions
# and `Cy` is the number of output channels (`y` and `z` can be the same array). Both `Cx` and
# `Cy` have to be an exact multiple of `group`.  `w` has size `(W...,Cx÷group,Cy)` where
# `(W...)` are the filter dimensions. `bias` has size `(1...,Cy,1)`.




using KernelAbstractions, CUDAKernels, Test, CUDA
using Test
if has_cuda_gpu()
    CUDA.allowscalar(false)
end



# Simple kernel for matrix multiplication
@kernel function matmul_kernel!(a, b, c)
  i, j = @index(Global, NTuple)

  # creating a temporary sum variable for matrix multiplication
  tmp_sum = zero(eltype(c))
  for k = 1:size(a)[2]
      tmp_sum += a[i,k] * b[k, j]
  end

  c[i,j] = tmp_sum
end

# Creating a wrapper kernel for launching with error checks
function matmul!(a, b, c)
  if size(a)[2] != size(b)[1]
      println("Matrix size mismatch!")
      return nothing
  end
  if isa(a, Array)
      kernel! = matmul_kernel!(CPU(),4)
  else
      kernel! = matmul_kernel!(KernelAbstractions.CUDADevice,256)
  end
  kernel!(a, b, c, ndrange=size(c)) 
end

a = rand(256,123)
b = rand(123, 45)
c = zeros(256, 45)

# beginning CPU tests, returns event
ev = matmul!(a,b,c)
wait(ev)

@test isapprox(c, a*b)

# beginning GPU tests
if has_cuda_gpu()
  d_a = CuArray(a)
  d_b = CuArray(b)
  d_c = CuArray(c)

  ev = matmul!(d_a, d_b, d_c)
  wait(ev)

  @test isapprox(Array(d_c), a*b)
end
KernelAbstractions.CUDADevice()


CUDA.versioninfo()