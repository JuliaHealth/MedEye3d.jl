
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