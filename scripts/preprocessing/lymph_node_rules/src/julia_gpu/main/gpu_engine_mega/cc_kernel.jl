module CcKernel

using KernelAbstractions
using Adapt

export cc_init_kernel!, cc_local_propagate!, cc_pointer_jump!, extract_largest_cc!

@kernel function cc_init_kernel!(labels, tensor, mask_id, mask_ch, x_dim, y_dim, z_dim)
    i, j, k = @index(Global, NTuple)
    if i <= x_dim && j <= y_dim && k <= z_dim
        val = (tensor[i, j, k, mask_ch] & (UInt32(1) << mask_id)) != 0
        if val
            labels[i, j, k] = Int32(i + (j-1)*x_dim + (k-1)*x_dim*y_dim)
        else
            labels[i, j, k] = Int32(0)
        end
    end
end

@kernel function cc_local_propagate!(labels_out, labels_in, tensor, mask_id, mask_ch, x_dim, y_dim, z_dim)
    i, j, k = @index(Global, NTuple)
    if i <= x_dim && j <= y_dim && k <= z_dim
        val = (tensor[i, j, k, mask_ch] & (UInt32(1) << mask_id)) != 0
        if val
            min_label = labels_in[i, j, k]
            
            # 6-connected neighbors only
            if i > 1
                n = (tensor[i-1, j, k, mask_ch] & (UInt32(1) << mask_id)) != 0
                if n
                    l = labels_in[i-1, j, k]
                    if l > 0 && l < min_label; min_label = l; end
                end
            end
            if i < x_dim
                n = (tensor[i+1, j, k, mask_ch] & (UInt32(1) << mask_id)) != 0
                if n
                    l = labels_in[i+1, j, k]
                    if l > 0 && l < min_label; min_label = l; end
                end
            end
            if j > 1
                n = (tensor[i, j-1, k, mask_ch] & (UInt32(1) << mask_id)) != 0
                if n
                    l = labels_in[i, j-1, k]
                    if l > 0 && l < min_label; min_label = l; end
                end
            end
            if j < y_dim
                n = (tensor[i, j+1, k, mask_ch] & (UInt32(1) << mask_id)) != 0
                if n
                    l = labels_in[i, j+1, k]
                    if l > 0 && l < min_label; min_label = l; end
                end
            end
            if k > 1
                n = (tensor[i, j, k-1, mask_ch] & (UInt32(1) << mask_id)) != 0
                if n
                    l = labels_in[i, j, k-1]
                    if l > 0 && l < min_label; min_label = l; end
                end
            end
            if k < z_dim
                n = (tensor[i, j, k+1, mask_ch] & (UInt32(1) << mask_id)) != 0
                if n
                    l = labels_in[i, j, k+1]
                    if l > 0 && l < min_label; min_label = l; end
                end
            end
            
            labels_out[i, j, k] = min_label
        else
            labels_out[i, j, k] = Int32(0)
        end
    end
end

@kernel function cc_pointer_jump!(labels_out, labels_in, tensor, mask_id, mask_ch, x_dim, y_dim, z_dim)
    i, j, k = @index(Global, NTuple)
    if i <= x_dim && j <= y_dim && k <= z_dim
        val = (tensor[i, j, k, mask_ch] & (UInt32(1) << mask_id)) != 0
        if val
            label = labels_in[i, j, k]
            if label > 0
                lk = (label - 1) ÷ (x_dim * y_dim) + 1
                remainder = (label - 1) % (x_dim * y_dim)
                lj = remainder ÷ x_dim + 1
                li = remainder % x_dim + 1
                
                if li >= 1 && li <= x_dim && lj >= 1 && lj <= y_dim && lk >= 1 && lk <= z_dim
                    labels_out[i, j, k] = labels_in[li, lj, lk]
                else
                    labels_out[i, j, k] = label
                end
            else
                labels_out[i, j, k] = Int32(0)
            end
        else
            labels_out[i, j, k] = Int32(0)
        end
    end
end

@kernel function extract_largest_cc!(tensor, labels, target_label, out_id, out_ch, x_dim, y_dim, z_dim)
    i, j, k = @index(Global, NTuple)
    if i <= x_dim && j <= y_dim && k <= z_dim
        if labels[i, j, k] == target_label
            tensor[i, j, k, out_ch] |= (UInt32(1) << out_id)
        end
    end
end

end
