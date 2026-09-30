function execute_mega2_hull_bridging(batch_json_str)
    batch_dict = JSON.parse(batch_json_str)
    in_ch1 = Int32(batch_dict["in_ch1"])
    in_id1 = UInt16(batch_dict["in_id1"])
    in_ch2 = Int32(batch_dict["in_ch2"])
    in_id2 = UInt16(batch_dict["in_id2"])
    out_ch = Int32(batch_dict["out_ch"])
    
    dims = TM_MEGA2.dims
    
    slice_present = adapt(TM_MEGA2.backend, zeros(Int32, 2, dims[3]))
    
    @kernel function check_slice!(slice_present, tensor_in, tensor_out, ch1, id1, ch2, id2, dims)
        I, J, K = @index(Global, NTuple)
        if I <= dims[1] && J <= dims[2] && K <= dims[3]
            v1 = ch1 > 0 ? (tensor_in[ch1, I, J, K] == id1) : (tensor_out[-ch1, I, J, K] > 0)
            v2 = ch2 > 0 ? (tensor_in[ch2, I, J, K] == id2) : (tensor_out[-ch2, I, J, K] > 0)
            if v1
                slice_present[1, K] = 1
            end
            if v2
                slice_present[2, K] = 1
            end
        end
    end
    
    @kernel function union_conditional!(tensor_out, tensor_in, slice_present, ch1, id1, ch2, id2, out_c, dims)
        I, J, K = @index(Global, NTuple)
        if I <= dims[1] && J <= dims[2] && K <= dims[3]
            if slice_present[1, K] > 0 && slice_present[2, K] > 0
                v1 = ch1 > 0 ? (tensor_in[ch1, I, J, K] == id1) : (tensor_out[-ch1, I, J, K] > 0)
                v2 = ch2 > 0 ? (tensor_in[ch2, I, J, K] == id2) : (tensor_out[-ch2, I, J, K] > 0)
                if v1 || v2
                    tensor_out[out_c, I, J, K] = UInt8(1)
                else
                    tensor_out[out_c, I, J, K] = UInt8(0)
                end
            else
                tensor_out[out_c, I, J, K] = UInt8(0)
            end
        end
    end

    backend = KernelAbstractions.get_backend(TM_MEGA2.tensor_in)
    check_slice!(backend, 256)(
        slice_present, TM_MEGA2.tensor_in, TM_MEGA2.tensor_out,
        in_ch1, in_id1, in_ch2, in_id2, dims, ndrange=dims)
    KernelAbstractions.synchronize(backend)
    
    union_conditional!(backend, 256)(
        TM_MEGA2.tensor_out, TM_MEGA2.tensor_in, slice_present,
        in_ch1, in_id1, in_ch2, in_id2, out_ch, dims, ndrange=dims)
    KernelAbstractions.synchronize(backend)
    
    planes_table, num_planes = HullPlanes.compute_sector_planes_mega_gpu(
        backend, TM_MEGA2.tensor_out, out_ch, dims)
    
    max_p = Int32(size(planes_table, 1))
    HullPlanes.fill_hull_gpu!(backend, 256)(
        TM_MEGA2.tensor_out, out_ch, planes_table, num_planes,
        Int32(dims[1]), Int32(dims[2]), Int32(dims[3]), max_p, ndrange=dims)
    KernelAbstractions.synchronize(backend)
    
    return "Executed MegaV2 hull bridging"
end
