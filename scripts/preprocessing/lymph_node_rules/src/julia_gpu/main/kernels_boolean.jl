using KernelAbstractions
using KernelAbstractions: @index, @localmem, @synchronize

@kernel function binary_op_kernel!(output, base_mask, other_mask, op)
    I, J, K = @index(Global, NTuple)
    dims = size(output)
    if I <= dims[1] && J <= dims[2] && K <= dims[3]
        v1 = base_mask[I, J, K] > 0
        v2 = other_mask[I, J, K] > 0
        
        if op == 1 # Intersect
            output[I, J, K] = (v1 && v2) ? UInt8(1) : UInt8(0)
        elseif op == 2 # Union
            output[I, J, K] = (v1 || v2) ? UInt8(1) : UInt8(0)
        elseif op == 3 # Subtract
            output[I, J, K] = (v1 && !v2) ? UInt8(1) : UInt8(0)
        end
    end
end

function run_boolean_op!(backend, output, base_mask, other_mask, op_str)
    op = 0
    op_lower = lowercase(op_str)
    if op_lower == "intersect" || op_lower == "intersection"
        op = 1
    elseif op_lower == "union"
        op = 2
    elseif op_lower == "subtract" || op_lower == "subtraction" || op_lower == "difference"
        op = 3
    else
        error("Unknown boolean op: $op_str")
    end
    
    kernel! = binary_op_kernel!(backend)
    kernel!(output, base_mask, other_mask, op, ndrange=size(output))
    KernelAbstractions.synchronize(backend)
end

run_bitwise_or!(backend, output, base_mask, other_mask) = run_boolean_op!(backend, output, base_mask, other_mask, "union")
run_bitwise_and!(backend, output, base_mask, other_mask) = run_boolean_op!(backend, output, base_mask, other_mask, "intersect")
