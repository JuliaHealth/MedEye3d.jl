using KernelAbstractions
using Adapt
using CUDA

@kernel function bbox_min_max_kernel!(out_bounds, @Const(tensor_out), ch, dims)
    I, J, K = @index(Global, NTuple)
    if I <= dims[1] && J <= dims[2] && K <= dims[3]
        if tensor_out[ch, I, J, K] > 0
            # out_bounds: [min_x, max_x, min_y, max_y, min_z, max_z]
            KernelAbstractions.@atomic min(out_bounds[1], Int32(I))
            KernelAbstractions.@atomic max(out_bounds[2], Int32(I))
            KernelAbstractions.@atomic min(out_bounds[3], Int32(J))
            KernelAbstractions.@atomic max(out_bounds[4], Int32(J))
            KernelAbstractions.@atomic min(out_bounds[5], Int32(K))
            KernelAbstractions.@atomic max(out_bounds[6], Int32(K))
        end
    end
end

function get_tensor_out_bboxes(backend, tensor_out, out_channels, dims)
    # out_channels is a list of channel indices (1-based)
    bboxes = []
    for ch in out_channels
        # Initialize bounds: min=Inf, max=-Inf
        bounds_cpu = Int32[dims[1]+1, -1, dims[2]+1, -1, dims[3]+1, -1]
        bounds_dev = adapt(backend, bounds_cpu)
        
        bbox_min_max_kernel!(backend, 256)(bounds_dev, tensor_out, Int32(ch), dims, ndrange=dims)
        KernelAbstractions.synchronize(backend)
        
        b = Array(bounds_dev)
        if b[1] <= b[2] # found something
            push!(bboxes, (b[1], b[2], b[3], b[4], b[5], b[6]))
        else
            push!(bboxes, (0, 0, 0, 0, 0, 0)) # empty
        end
    end
    return bboxes
end
