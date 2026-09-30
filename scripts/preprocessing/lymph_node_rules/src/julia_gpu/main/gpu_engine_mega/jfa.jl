using KernelAbstractions
using KernelAbstractions: @index
using CUDA

@kernel function jfa_init_kernel!(seeds, mask, dims)
    I, J, K = @index(Global, NTuple)
    if I <= dims[1] && J <= dims[2] && K <= dims[3]
        if mask[I, J, K] > 0
            seeds[1, I, J, K] = I
            seeds[2, I, J, K] = J
            seeds[3, I, J, K] = K
        else
            seeds[1, I, J, K] = 0
            seeds[2, I, J, K] = 0
            seeds[3, I, J, K] = 0
        end
    end
end

@kernel function jfa_step_kernel!(out_seeds, in_seeds, dims, step, sp_x, sp_y, sp_z)
    I, J, K = @index(Global, NTuple)
    if I <= dims[1] && J <= dims[2] && K <= dims[3]
        
        best_x = in_seeds[1, I, J, K]
        best_y = in_seeds[2, I, J, K]
        best_z = in_seeds[3, I, J, K]
        
        best_dist = 1f10
        if best_x > 0
            dx = (I - best_x) * sp_x
            dy = (J - best_y) * sp_y
            dz = (K - best_z) * sp_z
            best_dist = dx*dx + dy*dy + dz*dz
        end
        
        for k_off in -1:1
            for j_off in -1:1
                for i_off in -1:1
                    if i_off == 0 && j_off == 0 && k_off == 0
                        continue
                    end
                    
                    ni = I + i_off * step
                    nj = J + j_off * step
                    nk = K + k_off * step
                    
                    if ni >= 1 && ni <= dims[1] && nj >= 1 && nj <= dims[2] && nk >= 1 && nk <= dims[3]
                        cand_x = in_seeds[1, ni, nj, nk]
                        cand_y = in_seeds[2, ni, nj, nk]
                        cand_z = in_seeds[3, ni, nj, nk]
                        
                        if cand_x > 0
                            dx = (I - cand_x) * sp_x
                            dy = (J - cand_y) * sp_y
                            dz = (K - cand_z) * sp_z
                            cand_dist = dx*dx + dy*dy + dz*dz
                            
                            if cand_dist < best_dist
                                best_dist = cand_dist
                                best_x = cand_x
                                best_y = cand_y
                                best_z = cand_z
                            end
                        end
                    end
                end
            end
        end
        
        out_seeds[1, I, J, K] = best_x
        out_seeds[2, I, J, K] = best_y
        out_seeds[3, I, J, K] = best_z
    end
end

function run_jfa_3d(backend, mask_device, sp_x, sp_y, sp_z)
    dims = size(mask_device)
    
    seeds_A = TM_MEGA2.jfa_seeds_A
    seeds_B = TM_MEGA2.jfa_seeds_B
    
    jfa_init_kernel!(backend, 256)(seeds_A, mask_device, dims, ndrange=dims)
    KernelAbstractions.synchronize(backend)
    
    max_dim = max(dims[1], dims[2], dims[3])
    step = 1
    while step < max_dim
        step *= 2
    end
    step = step ÷ 2
    
    current_in = seeds_A
    current_out = seeds_B
    
    while step >= 1
        jfa_step_kernel!(backend, 256)(current_out, current_in, dims, step, Float32(sp_x), Float32(sp_y), Float32(sp_z), ndrange=dims)
        KernelAbstractions.synchronize(backend)
        
        step = step ÷ 2
        
        temp = current_in
        current_in = current_out
        current_out = temp
    end
    
    return current_in
end
