module VMKernel

using Adapt
using KernelAbstractions
using CUDA
using ..VMOpcodes

export execute_megakernel_level!, refine_inguinal_kernel!, split_packed_mask_kernel!, zero_overlap_precedence_kernel!, fused_overlap_precedence_kernel!, split_neck_level_2_kernel!, axillary_2_3_reassign_kernel!

@kernel function megakernel_dag_level!(
    level_output::AbstractArray{UInt8, 4},
    @Const(packed_data),
    @Const(instructions),
    num_instructions::Int32,
    dims_x::Int32, dims_y::Int32, dims_z::Int32,
    sp_x::Float32, sp_y::Float32, sp_z::Float32
)
    i, j, k = @index(Global, NTuple)
    
    if i <= dims_x && j <= dims_y && k <= dims_z
        active = false
        curr_rule = Int32(-1)
        
        for inst_idx in Int32(1):num_instructions
            inst = instructions[inst_idx]
            op = inst.opcode
            rule = inst.rule_idx
            
            if rule != curr_rule
                curr_rule = rule
                active = false
            end
            
            if op == OP_LOAD_OUTPUT_BUFFER
                active = (level_output[i, j, k, rule] > UInt8(0))

            elseif op == OP_LOAD_BASE_MASK
                ch = inst.arg1
                id = UInt8(inst.arg2)
                active = (packed_data[i, j, k, ch] == id)
                
            elseif op == OP_UNION_BASE_MASK
                if !active
                    ch = inst.arg1
                    id = UInt8(inst.arg2)
                    active = (packed_data[i, j, k, ch] == id)
                end
                
            elseif op == OP_DILATE_BOX
                if !active
                    ch = inst.arg1
                    id = UInt8(inst.arg2)
                    rx = inst.arg3
                    ry = inst.arg4
                    rz = inst.arg5
                    
                    min_x = max(Int32(1), i - rx)
                    max_x = min(dims_x, i + rx)
                    min_y = max(Int32(1), j - ry)
                    max_y = min(dims_y, j + ry)
                    min_z = max(Int32(1), k - rz)
                    max_z = min(dims_z, k + rz)
                    
                    found = false
                    zz = min_z
                    while zz <= max_z && !found
                        yy = min_y
                        while yy <= max_y && !found
                            xx = min_x
                            while xx <= max_x && !found
                                if packed_data[xx, yy, zz, ch] == id
                                    found = true
                                end
                                xx += Int32(1)
                            end
                            yy += Int32(1)
                        end
                        zz += Int32(1)
                    end
                    active = found
                end

            elseif op == OP_ANISOTROPIC_EXPAND
                if !active
                    # arg3..arg8 = base mask ROI [min_i, max_i, min_j, max_j, min_k, max_k]
                    # arg9..arg14 = directional radii [rad_x_pos, rad_x_neg, rad_y_pos, rad_y_neg, rad_z_pos, rad_z_neg]
                    # Check if query voxel (i, j, k) is within expanded ROI bounds
                    if i >= inst.arg3 - inst.arg10 && i <= inst.arg4 + inst.arg9 &&
                       j >= inst.arg5 - inst.arg12 && j <= inst.arg6 + inst.arg11 &&
                       k >= inst.arg7 - inst.arg14 && k <= inst.arg8 + inst.arg13
                        
                        ch = inst.arg1
                        id = UInt8(inst.arg2)
                        if packed_data[i, j, k, ch] == id
                            active = true
                        else
                            is_hit = false
                            # Candidate mask voxel (cx, cy, cz) is constrained to intersection of base mask ROI and neighborhood
                            start_z = max(inst.arg7, k - inst.arg13)
                            end_z   = min(inst.arg8, k + inst.arg14)
                            
                            start_y = max(inst.arg5, j - inst.arg11)
                            end_y   = min(inst.arg6, j + inst.arg12)
                            
                            start_x = max(inst.arg3, i - inst.arg9)
                            end_x   = min(inst.arg4, i + inst.arg10)
                            
                            cz = start_z
                            while cz <= end_z && !is_hit
                                dz_vox = k - cz
                                dz = Float32(dz_vox) * sp_z
                                mz = dz >= 0f0 ? inst.farg5 : inst.farg6
                                
                                cy = start_y
                                while cy <= end_y && !is_hit
                                    dy_vox = j - cy
                                    dy = Float32(dy_vox) * sp_y
                                    my = dy >= 0f0 ? inst.farg3 : inst.farg4
                                    
                                    cx = start_x
                                    while cx <= end_x && !is_hit
                                        if packed_data[cx, cy, cz, ch] == id
                                            dx_vox = i - cx
                                            dx = Float32(dx_vox) * sp_x
                                            mx = dx >= 0f0 ? inst.farg1 : inst.farg2
                                            
                                            abs_dx = abs(dx)
                                            abs_dy = abs(dy)
                                            abs_dz = abs(dz)
                                            tot = abs_dx + abs_dy + abs_dz + 1f-8
                                            
                                            thresh = (abs_dx / tot) * mx + (abs_dy / tot) * my + (abs_dz / tot) * mz
                                            dist_sq = dx*dx + dy*dy + dz*dz
                                            
                                            if dist_sq <= thresh * thresh && thresh > 0f0
                                                is_hit = true
                                            end
                                        end
                                        cx += Int32(1)
                                    end # cx
                                    cy += Int32(1)
                                end # cy
                                cz += Int32(1)
                            end # cz
                            if is_hit
                                active = true
                            end
                        end
                    end
                end
                
            elseif op == OP_Z_BOUNDS
                if active
                    k_min = inst.arg1
                    k_max = inst.arg2
                    if k < k_min || k > k_max
                        active = false
                    end
                end
                
            elseif op == OP_BBOX
                if active
                    axis = inst.arg1 # 1=x, 2=y, 3=z
                    min_val = inst.arg2
                    max_val = inst.arg3
                    val = (axis == Int32(1)) ? i : ((axis == Int32(2)) ? j : k)
                    if val < min_val || val > max_val
                        active = false
                    end
                end
                
            elseif op == OP_PLANE
                if active
                    side = inst.arg1 # 1 = positive, -1 = negative
                    nx = inst.farg1
                    ny = inst.farg2
                    nz = inst.farg3
                    d  = inst.farg4
                    val = nx * Float32(i) + ny * Float32(j) + nz * Float32(k) + d
                    if side == Int32(1)
                        if val < 0.0f0; active = false; end
                    else
                        if val > 0.0f0; active = false; end
                    end
                end
                
            elseif op == OP_SLICE_CONSTRAINT
                if active
                    ch = inst.arg1
                    id = UInt8(inst.arg2)
                    c_type = inst.arg3 # 0=Sup, 1=Inf, 2=Ant, 3=Post, 4=Right, 5=Left
                    bound_val = inst.arg5
                    is_slice_wise = (inst.arg4 == Int32(1))
                    
                    failed = false
                    if !is_slice_wise
                        if c_type == Int32(0) && k <= bound_val
                            failed = true
                        elseif c_type == Int32(1) && k >= bound_val
                            failed = true
                        elseif c_type == Int32(2) && j <= bound_val
                            failed = true
                        elseif c_type == Int32(3) && j >= bound_val
                            failed = true
                        elseif c_type == Int32(4) && i <= bound_val
                            failed = true
                        elseif c_type == Int32(5) && i >= bound_val
                            failed = true
                        end
                    else
                        # slice-wise constraint: check along constant-k plane
                        if c_type == Int32(2) # AnteriorTo (must be anterior: no landmark voxel at or anterior to j)
                            yy = Int32(1)
                            while yy <= j && !failed
                                xx = Int32(1)
                                while xx <= dims_x && !failed
                                    if packed_data[xx, yy, k, ch] == id
                                        failed = true
                                    end
                                    xx += Int32(1)
                                end
                                yy += Int32(1)
                            end
                        elseif c_type == Int32(3) # PosteriorTo
                            yy = j
                            while yy <= dims_y && !failed
                                xx = Int32(1)
                                while xx <= dims_x && !failed
                                    if packed_data[xx, yy, k, ch] == id
                                        failed = true
                                    end
                                    xx += Int32(1)
                                end
                                yy += Int32(1)
                            end
                        elseif c_type == Int32(4) # RightOf
                            xx = Int32(1)
                            while xx <= i && !failed
                                if packed_data[xx, j, k, ch] == id
                                    failed = true
                                end
                                xx += Int32(1)
                            end
                        elseif c_type == Int32(5) # LeftOf
                            xx = i
                            while xx <= dims_x && !failed
                                if packed_data[xx, j, k, ch] == id
                                    failed = true
                                end
                                xx += Int32(1)
                            end
                        end
                    end
                    
                    if failed
                        active = false
                    end
                end
                
            elseif op == OP_EXCLUDE_MASK
                if active
                    ch = inst.arg1
                    id = UInt8(inst.arg2)
                    if packed_data[i, j, k, ch] == id
                        active = false
                    end
                end
                
            elseif op == OP_EXCLUDE_MARGIN
                if active
                    ch = inst.arg1
                    id = UInt8(inst.arg2)
                    rx = inst.arg3
                    ry = inst.arg4
                    rz = inst.arg5
                    
                    min_x = max(Int32(1), i - rx)
                    max_x = min(dims_x, i + rx)
                    min_y = max(Int32(1), j - ry)
                    max_y = min(dims_y, j + ry)
                    min_z = max(Int32(1), k - rz)
                    max_z = min(dims_z, k + rz)
                    
                    found = false
                    zz = min_z
                    while zz <= max_z && !found
                        yy = min_y
                        while yy <= max_y && !found
                            xx = min_x
                            while xx <= max_x && !found
                                if packed_data[xx, yy, zz, ch] == id
                                    found = true
                                end
                                xx += Int32(1)
                            end
                            yy += Int32(1)
                        end
                        zz += Int32(1)
                    end
                    if found
                        active = false
                    end
                end
                
            elseif op == OP_INTERSECT_MASK
                # Intersection: deactivate if mask NOT present at this voxel
                if active
                    ch = inst.arg1
                    id = UInt8(inst.arg2)
                    if packed_data[i, j, k, ch] != id
                        active = false
                    end
                end
                
            elseif op == OP_WRITE_OUTPUT
                level_output[i, j, k, rule] = active ? UInt8(1) : UInt8(0)
                active = false
            end
        end
    end
end

function execute_megakernel_level!(
    backend,
    level_output,
    packed_data,
    instructions::Vector{VMInstruction};
    spacing::Tuple{Float64, Float64, Float64}=(1.0, 1.0, 1.0)
)
    if isempty(instructions)
        return
    end
    dims = size(packed_data)[1:3]
    num_inst = Int32(length(instructions))
    inst_gpu = adapt(backend, instructions)
    sp_x, sp_y, sp_z = Float32(spacing[1]), Float32(spacing[2]), Float32(spacing[3])
    
    kernel! = megakernel_dag_level!(backend)
    kernel!(level_output, packed_data, inst_gpu, num_inst,
            Int32(dims[1]), Int32(dims[2]), Int32(dims[3]),
            sp_x, sp_y, sp_z,
            ndrange=dims)
    KernelAbstractions.synchronize(backend)
end


@kernel function refine_inguinal_kernel!(
    packed_data,
    ch::Int32, id::UInt8,
    p1_x::Float32, p1_y::Float32, p2_x::Float32, p2_y::Float32,
    dims_x::Int32, dims_y::Int32, dims_z::Int32
)
    i, j, k = @index(Global, NTuple)
    if i <= dims_x && j <= dims_y && k <= dims_z
        if packed_data[i, j, k, ch] == id
            y_line = p1_y + (p2_y - p1_y) * (Float32(i) - p1_x) / (p2_x - p1_x)
            if Float32(j) >= y_line
                packed_data[i, j, k, ch] = UInt8(0)
            end
        end
    end
end

@kernel function split_packed_mask_kernel!(
    @Const(packed_data),
    out_left, out_right,
    @Const(slice_split_x),
    ch::Int32, id::UInt8,
    dims_x::Int32, dims_y::Int32, dims_z::Int32
)
    i, j, k = @index(Global, NTuple)
    if i <= dims_x && j <= dims_y && k <= dims_z
        if packed_data[i, j, k, ch] == id
            sx = slice_split_x[k]
            if i < sx
                out_right[i, j, k] = UInt8(1)
            else
                out_left[i, j, k] = UInt8(1)
            end
        end
    end
end

@kernel function zero_overlap_precedence_kernel!(
    packed_data,
    ch_zero::Int32, id_zero::UInt8,
    ch_keep::Int32, id_keep::UInt8,
    ox::Int32, oy::Int32, oz::Int32,
    nx::Int32, ny::Int32, nz::Int32
)
    # The kernel is launched with ndrange = (nx, ny, nz)
    # so we just add the offset (ox, oy, oz) to get global indices
    i_, j_, k_ = @index(Global, NTuple)
    i = i_ + ox - Int32(1)
    j = j_ + oy - Int32(1)
    k = k_ + oz - Int32(1)
    
    @inbounds begin
        if packed_data[i, j, k, ch_zero] == id_zero && packed_data[i, j, k, ch_keep] == id_keep
            packed_data[i, j, k, ch_zero] = UInt8(0)
        end
    end
end

# Fused overlap resolution kernel: processes ALL precedence pairs in one launch.
# pair_data layout: (n_pairs, 10) where each row is:
#   [ch_zero, id_zero, ch_keep, id_keep, ox_min, ox_max, oy_min, oy_max, oz_min, oz_max]
# Launched with ndrange = (dims_x, dims_y, dims_z) over the FULL volume.
# Each voxel checks all pairs; bounding box test skips irrelevant pairs.
@kernel function fused_overlap_precedence_kernel!(
    packed_data,
    @Const(pair_ch_zero),    # Int32[n_pairs]
    @Const(pair_id_zero),    # UInt8[n_pairs]
    @Const(pair_ch_keep),    # Int32[n_pairs]
    @Const(pair_id_keep),    # UInt8[n_pairs]
    @Const(pair_ox_min),     # Int32[n_pairs]
    @Const(pair_ox_max),     # Int32[n_pairs]
    @Const(pair_oy_min),     # Int32[n_pairs]
    @Const(pair_oy_max),     # Int32[n_pairs]
    @Const(pair_oz_min),     # Int32[n_pairs]
    @Const(pair_oz_max),     # Int32[n_pairs]
    n_pairs::Int32
)
    i, j, k = @index(Global, NTuple)
    
    @inbounds for p in Int32(1):n_pairs
        # Bounding box check — skip if this voxel is outside the overlap region
        if i < pair_ox_min[p] || i > pair_ox_max[p]; continue; end
        if j < pair_oy_min[p] || j > pair_oy_max[p]; continue; end
        if k < pair_oz_min[p] || k > pair_oz_max[p]; continue; end
        
        # This voxel is inside the overlap bbox — check if both stations claim it
        ch_z = pair_ch_zero[p]
        id_z = pair_id_zero[p]
        ch_k = pair_ch_keep[p]
        id_k = pair_id_keep[p]
        
        if packed_data[i, j, k, ch_z] == id_z && packed_data[i, j, k, ch_k] == id_k
            packed_data[i, j, k, ch_z] = UInt8(0)
        end
    end
end
@kernel function split_neck_level_2_kernel!(
    @Const(packed_data),
    out_2a, out_2b,
    @Const(ijv_post_y_arr),
    ch::Int32, id::UInt8,
    dims_x::Int32, dims_y::Int32, dims_z::Int32
)
    i, j, k = @index(Global, NTuple)
    if i <= dims_x && j <= dims_y && k <= dims_z
        if packed_data[i, j, k, ch] == id
            ijv_y = ijv_post_y_arr[k]
            if ijv_y > 0
                # In RAS (+Y is Anterior), larger j is anterior.
                # IIa is anterior, IIb is posterior.
                # So j >= ijv_y should be IIa!
                if j <= ijv_y
                    out_2a[i, j, k] = UInt8(1)
                else
                    out_2b[i, j, k] = UInt8(1)
                end
            else
                out_2a[i, j, k] = UInt8(1)
                out_2b[i, j, k] = UInt8(1)
            end
        end
    end
end

@kernel function axillary_2_3_reassign_kernel!(out2, out3, in2, in3, pec_x, is_left, dims_z)
    i, j, k = @index(Global, NTuple)
    
    if k <= dims_z
        x_val = pec_x[k]
        
        v2 = in2[i, j, k]
        v3 = in3[i, j, k]
        
        if x_val > 0 && v2 > 0
            is_medial = is_left ? (i < x_val) : (i > x_val)
            if is_medial
                v2 = UInt8(0)
                v3 = UInt8(1)
            end
        end
        
        out2[i, j, k] = v2
        out3[i, j, k] = v3
    end
end
end # module