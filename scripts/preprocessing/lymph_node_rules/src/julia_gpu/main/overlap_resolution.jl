using ImageMorphology


using CUDA, KernelAbstractions
using .MaskPacker: PackedTensor, unpack_mask

function has_precedence(name1::String, name2::String, rules::Dict)
    # A rule's `overlap_precedence` list contains the masks that take precedence OVER that rule.
    # Therefore, name1 has precedence over name2 IF name1 (or base1) is in name2's overlap_precedence.
    b1 = replace(name1, r"(_Left|_Right|_left|_right|_L|_R)$" => "")
    b2 = replace(name2, r"(_Left|_Right|_left|_right|_L|_R)$" => "")
    
    prec2 = get(get(rules, b2, Dict()), "overlap_precedence", [])
    prec2_full = get(get(rules, name2, Dict()), "overlap_precedence", [])
    
    if (b1 in prec2) || (name1 in prec2) || (b1 in prec2_full) || (name1 in prec2_full) || get(get(rules, name1, Dict()), "is_dominant", false)
        return true
    end
    return false
end

const CANONICAL_LN_STATIONS_LOWER = Set([
    # Neck
    "neck_level_ia_submental",
    "neck_level_ib_submandibular_left", "neck_level_ib_submandibular_right",
    "neck_level_iia_upper_jugular_left", "neck_level_iia_upper_jugular_right",
    "neck_level_iib_upper_jugular_left", "neck_level_iib_upper_jugular_right",
    "neck_level_iii_middle_jugular_left", "neck_level_iii_middle_jugular_right",
    "neck_level_iv_lower_jugular_left", "neck_level_iv_lower_jugular_right",
    "neck_level_va_upper_posterior_triangle_left", "neck_level_va_upper_posterior_triangle_right",
    "neck_level_vi_anterior_cervical",
    "neck_level_xb_occipital_left", "neck_level_xb_occipital_right",
    "neck_parotid_left", "neck_parotid_right",
    "neck_retropharyngeal",
    # Thoracic
    "thoracic_station_2_upperparatracheal_left", "thoracic_station_2_upperparatracheal_right",
    "thoracic_station_3a_prevascular_left", "thoracic_station_3a_prevascular_right",
    "thoracic_station_3p_retrotracheal_left", "thoracic_station_3p_retrotracheal_right",
    "thoracic_station_4_lowerparatracheal_left", "thoracic_station_4_lowerparatracheal_right",
    "thoracic_station_5_subaortic_left",
    "thoracic_station_6_paraaortic",
    "thoracic_station_7_subcarinial",
    "thoracic_station_8_paraoesophageal_left", "thoracic_station_8_paraoesophageal_right",
    "thoracic_station_hilar_interlobar_left", "thoracic_station_hilar_interlobar_right",
    "thoracic_station_prepericardial_left", "thoracic_station_prepericardial_right",
    "thoracic_mammary_left", "thoracic_mammary_right",
    # Axillary
    "axillary_level_i_left", "axillary_level_i_right",
    "axillary_level_ii_left", "axillary_level_ii_right",
    "axillary_level_iii_left", "axillary_level_iii_right",
    "axillary_rotter_left", "axillary_rotter_right",
    # Abdominal - Gastric
    "abdominal_station_1_right_paracardial",
    "abdominal_station_2_left_paracardial",
    "abdominal_station_3_lesser_curvature",
    "abdominal_station_4_greater_curvature",
    "abdominal_station_5_suprapyloric",
    "abdominal_station_6_infrapyloric",
    "abdominal_station_7_left_gastric",
    "abdominal_station_8_common_hepatic",
    "abdominal_station_9_celiac",
    "abdominal_station_10_splenic_hilum",
    "abdominal_station_11_splenic_artery",
    "abdominal_station_13_posterior_pancreaticoduodenal",
    "abdominal_station_17_anterior_pancreaticoduodenal",
    "abdominal_station_14_sma",
    "abdominal_station_inferior_pancreatic",
    # Abdominal - Paraaortic
    "abdominal_station_16a1_aortic_hiatus",
    "abdominal_station_16a2_upper_middle_paraaortic",
    "abdominal_station_16b1_lower_middle_paraaortic",
    "abdominal_station_16b2_caudal_paraaortic",
    # Abdominal - Renal
    "abdominal_renal_hilar_left", "abdominal_renal_hilar_right",
    # Abdominal - Pelvic
    "abdominal_common_iliac_left", "abdominal_common_iliac_right",
    "abdominal_external_iliac_left", "abdominal_external_iliac_right",
    "abdominal_iliac_bifurcation_left", "abdominal_iliac_bifurcation_right",
    "abdominal_internal_iliac_left", "abdominal_internal_iliac_right",
    "abdominal_mesenteric_interenteric",
    "abdominal_obturator_left", "abdominal_obturator_right",
    "abdominal_pararectal",
    "abdominal_presacral",
    # Inguinal
    "deep_inguinal_left", "deep_inguinal_right",
    "superficial_inguinal_left", "superficial_inguinal_right",
])

@kernel function edt_assign_overlap_kernel!(
    @Const(overlap),
    dt1, dt2,
    out1, out2,
    dims_x, dims_y, dims_z
)
    i, j, k = @index(Global, NTuple)
    
    if i <= dims_x && j <= dims_y && k <= dims_z
        if overlap[i, j, k] > 0
            d1 = dt1[i, j, k]
            d2 = dt2[i, j, k]
            
            if d1 < d2
                out1[i, j, k] = 1
                out2[i, j, k] = 0
            elseif d2 < d1
                out1[i, j, k] = 0
                out2[i, j, k] = 1
            else
                # Tie: assign to out1 for determinism
                out1[i, j, k] = 1
                out2[i, j, k] = 0
            end
        end
    end
end

function resolve_overlaps_gpu!(packed_tensor::PackedTensor, rules::Dict; spacing::Tuple=(0.9765625, 0.9765625, 3.0), anatomy_masks::AbstractDict=Dict{String, Any}(), backend::KernelAbstractions.Backend=CUDA.functional() ? CUDABackend() : CPU())
    
    # 1. Gather keys from registry that are canonical clinical lymph node stations
    dims = size(packed_tensor.data)[1:3]
    ln_keys = String[]
    for k in keys(packed_tensor.registry)
        if lowercase(k) in CANONICAL_LN_STATIONS_LOWER
            push!(ln_keys, k)
        end
    end
    keys_sorted = sort(ln_keys)
    n = length(keys_sorted)
    println("  -> GPU Overlap resolution: $(n) canonical LN stations to check")
    
    # Allocate maximum potential bounding box for overlap EDT passes
    # Preallocate reusable buffers to avoid per-pair allocation/free overhead
    # First pass: find max bounding box intersection size (with padding)
    max_nx, max_ny, max_nz = 1, 1, 1
    for i in 1:n
        name1 = keys_sorted[i]
        bb1 = get(packed_tensor.bboxes, name1, nothing)
        if bb1 === nothing
            m1_buf, k1 = StaticArena.acquire_mask(backend)
            MaskPacker.unpack_mask!(m1_buf, packed_tensor, name1)
            bb1 = RuleExecutors.gpu_bounding_box(backend, m1_buf)
            StaticArena.release_mask(k1)
            if bb1 !== nothing
                packed_tensor.bboxes[name1] = bb1
            end
        end
    end
    # Estimate max overlap box size from all bounding boxes
    for i in 1:n
        bb1 = get(packed_tensor.bboxes, keys_sorted[i], nothing)
        bb1 === nothing && continue
        for j in (i+1):n
            bb2 = get(packed_tensor.bboxes, keys_sorted[j], nothing)
            bb2 === nothing && continue
            ox_min = max(1, max(bb1[1], bb2[1]) - 5)
            ox_max = min(dims[1], min(bb1[2], bb2[2]) + 5)
            oy_min = max(1, max(bb1[3], bb2[3]) - 5)
            oy_max = min(dims[2], min(bb1[4], bb2[4]) + 5)
            oz_min = max(1, max(bb1[5], bb2[5]) - 5)
            oz_max = min(dims[3], min(bb1[6], bb2[6]) + 5)
            if ox_min <= ox_max && oy_min <= oy_max && oz_min <= oz_max
                max_nx = max(max_nx, ox_max - ox_min + 1)
                max_ny = max(max_ny, oy_max - oy_min + 1)
                max_nz = max(max_nz, oz_max - oz_min + 1)
            end
        end
    end
    # Preallocate reusable EDT buffers at max size
    _dt1_buf = KernelAbstractions.allocate(backend, Float32, (max_nx, max_ny, max_nz))
    _dt2_buf = KernelAbstractions.allocate(backend, Float32, (max_nx, max_ny, max_nz))
    println("  -> Preallocated overlap EDT buffers: $(max_nx)×$(max_ny)×$(max_nz) Float32")
    
    # --- Phase 1: Collect all precedence pairs and run ONE fused kernel ---
    prec_ch_zero = Int32[]
    prec_id_zero = UInt8[]
    prec_ch_keep = Int32[]
    prec_id_keep = UInt8[]
    prec_ox_min = Int32[]
    prec_ox_max = Int32[]
    prec_oy_min = Int32[]
    prec_oy_max = Int32[]
    prec_oz_min = Int32[]
    prec_oz_max = Int32[]
    
    # Also collect distance-based pairs for Phase 2
    distance_pairs = Tuple{String, String, Int, UInt8, Int, UInt8, NTuple{6,Int}}[]
    
    for i in 1:n
        name1 = keys_sorted[i]
        ch1, id1 = packed_tensor.registry[name1]
        bb1 = get(packed_tensor.bboxes, name1, nothing)
        bb1 === nothing && continue
        
        for j in (i+1):n
            name2 = keys_sorted[j]
            ch2, id2 = packed_tensor.registry[name2]
            bb2 = get(packed_tensor.bboxes, name2, nothing)
            bb2 === nothing && continue
            
            # Fast Bounding Box Intersection Check
            ox_min = max(bb1[1], bb2[1])
            ox_max = min(bb1[2], bb2[2])
            oy_min = max(bb1[3], bb2[3])
            oy_max = min(bb1[4], bb2[4])
            oz_min = max(bb1[5], bb2[5])
            oz_max = min(bb1[6], bb2[6])
            
            if ox_min > ox_max || oy_min > oy_max || oz_min > oz_max
                continue
            end
            
            prec1 = has_precedence(name1, name2, rules)
            prec2 = has_precedence(name2, name1, rules)
            
            if prec1 && !prec2
                # name1 has precedence, zero out name2 where both overlap
                push!(prec_ch_zero, Int32(ch2)); push!(prec_id_zero, UInt8(id2))
                push!(prec_ch_keep, Int32(ch1)); push!(prec_id_keep, UInt8(id1))
                push!(prec_ox_min, Int32(ox_min)); push!(prec_ox_max, Int32(ox_max))
                push!(prec_oy_min, Int32(oy_min)); push!(prec_oy_max, Int32(oy_max))
                push!(prec_oz_min, Int32(oz_min)); push!(prec_oz_max, Int32(oz_max))
            elseif prec2 && !prec1
                # name2 has precedence, zero out name1
                push!(prec_ch_zero, Int32(ch1)); push!(prec_id_zero, UInt8(id1))
                push!(prec_ch_keep, Int32(ch2)); push!(prec_id_keep, UInt8(id2))
                push!(prec_ox_min, Int32(ox_min)); push!(prec_ox_max, Int32(ox_max))
                push!(prec_oy_min, Int32(oy_min)); push!(prec_oy_max, Int32(oy_max))
                push!(prec_oz_min, Int32(oz_min)); push!(prec_oz_max, Int32(oz_max))
            else
                # Distance tiebreaker — collect for Phase 2
                push!(distance_pairs, (name1, name2, ch1, id1, ch2, id2, (ox_min, ox_max, oy_min, oy_max, oz_min, oz_max)))
            end
        end
    end
    
    # Launch ONE fused kernel for all precedence pairs
    n_prec = length(prec_ch_zero)
    if n_prec > 0
        t_prec_start = time()
        gpu_ch_zero = adapt(backend, prec_ch_zero)
        gpu_id_zero = adapt(backend, prec_id_zero)
        gpu_ch_keep = adapt(backend, prec_ch_keep)
        gpu_id_keep = adapt(backend, prec_id_keep)
        gpu_ox_min = adapt(backend, prec_ox_min)
        gpu_ox_max = adapt(backend, prec_ox_max)
        gpu_oy_min = adapt(backend, prec_oy_min)
        gpu_oy_max = adapt(backend, prec_oy_max)
        gpu_oz_min = adapt(backend, prec_oz_min)
        gpu_oz_max = adapt(backend, prec_oz_max)
        
        fused_kernel! = fused_overlap_precedence_kernel!(backend)
        fused_kernel!(
            packed_tensor.data,
            gpu_ch_zero, gpu_id_zero, gpu_ch_keep, gpu_id_keep,
            gpu_ox_min, gpu_ox_max, gpu_oy_min, gpu_oy_max, gpu_oz_min, gpu_oz_max,
            Int32(n_prec),
            ndrange=dims
        )
        KernelAbstractions.synchronize(backend)
        t_prec_end = time()
        println("  -> Fused precedence kernel: $(n_prec) pairs resolved in $(round(t_prec_end - t_prec_start, digits=3))s (1 kernel launch)")
    end
    
    # --- Phase 2: Distance-based EDT for remaining pairs ---
    println("  -> Distance-based EDT: $(length(distance_pairs)) pairs to resolve")
    for (name1, name2, ch1, id1, ch2, id2, bbox) in distance_pairs
            ox_min, ox_max, oy_min, oy_max, oz_min, oz_max = bbox
                # Distance tiebreaker (requires unpacking bounding box)
                # In python, the bounding box is padded by 5
                p_ox_min = max(1, ox_min - 5)
                p_ox_max = min(dims[1], ox_max + 5)
                p_oy_min = max(1, oy_min - 5)
                p_oy_max = min(dims[2], oy_max + 5)
                p_oz_min = max(1, oz_min - 5)
                p_oz_max = min(dims[3], oz_max + 5)
                
                v_ch1 = view(packed_tensor.data, p_ox_min:p_ox_max, p_oy_min:p_oy_max, p_oz_min:p_oz_max, ch1)
                v_ch2 = view(packed_tensor.data, p_ox_min:p_ox_max, p_oy_min:p_oy_max, p_oz_min:p_oz_max, ch2)
                
                sub1 = v_ch1 .== id1
                sub2 = v_ch2 .== id2
                sub_ov = sub1 .& sub2
                
                if !any(sub_ov)
                    continue
                end
                
                clean1 = sub1 .& (.~sub_ov)
                clean2 = sub2 .& (.~sub_ov)
                
                if !any(clean1)
                    # name1 fully contained, use centroid of sub1 as seed
                    cx, cy, cz = 0, 0, 0
                    pts = adapt(Array, findall(sub1))
                    if !isempty(pts)
                        for pt in pts
                            cx += pt[1]; cy += pt[2]; cz += pt[3]
                        end
                        cx = round(Int, cx/length(pts))
                        cy = round(Int, cy/length(pts))
                        cz = round(Int, cz/length(pts))
                        c_arr = zeros(Bool, size(clean1))
                        c_arr[cx, cy, cz] = true
                        clean1 .|= adapt(typeof(clean1), c_arr)
                    end
                end
                if !any(clean2)
                    # name2 fully contained, use centroid of sub2 as seed
                    cx, cy, cz = 0, 0, 0
                    pts = adapt(Array, findall(sub2))
                    if !isempty(pts)
                        for pt in pts
                            cx += pt[1]; cy += pt[2]; cz += pt[3]
                        end
                        cx = round(Int, cx/length(pts))
                        cy = round(Int, cy/length(pts))
                        cz = round(Int, cz/length(pts))
                        c_arr = zeros(Bool, size(clean2))
                        c_arr[cx, cy, cz] = true
                        clean2 .|= adapt(typeof(clean2), c_arr)
                    end
                end
                
                println("  -> Distance-based split on GPU: $(name1) vs $(name2)")
                
                seed1_gpu = map(x -> x ? UInt8(1) : UInt8(0), clean1)
                seed2_gpu = map(x -> x ? UInt8(1) : UInt8(0), clean2)
                
                crop_dims = size(clean1)
                # Use preallocated buffers via views (guaranteed to fit since we computed max_nx/ny/nz)
                dt1_gpu = view(_dt1_buf, 1:crop_dims[1], 1:crop_dims[2], 1:crop_dims[3])
                dt2_gpu = view(_dt2_buf, 1:crop_dims[1], 1:crop_dims[2], 1:crop_dims[3])
                
                gpu_exact_edt!(dt1_gpu, backend, seed1_gpu, spacing)
                gpu_exact_edt!(dt2_gpu, backend, seed2_gpu, spacing)
                
                out1_gpu = copy(seed1_gpu)
                out2_gpu = copy(seed2_gpu)
                
                edt_kernel! = edt_assign_overlap_kernel!(backend)
                edt_kernel!(
                    sub_ov, dt1_gpu, dt2_gpu, out1_gpu, out2_gpu,
                    Int32(crop_dims[1]), Int32(crop_dims[2]), Int32(crop_dims[3]),
                    ndrange=crop_dims
                )
                KernelAbstractions.synchronize(backend)
                
                # Write back resolved voxels (only for the overlap region to avoid disturbing non-overlap)
                v_ch1 .= ifelse.(sub_ov, ifelse.(out1_gpu .> 0, id1, UInt8(0)), v_ch1)
                v_ch2 .= ifelse.(sub_ov, ifelse.(out2_gpu .> 0, id2, UInt8(0)), v_ch2)
    end
    
    # Free preallocated EDT buffers
    if backend isa CUDABackend
        try; CUDA.unsafe_free!(_dt1_buf); catch; end
        try; CUDA.unsafe_free!(_dt2_buf); catch; end
    end
    
    # Finally, extract all resolved masks to CPU BitArrays for saving
    println("  -> Extracting resolved masks to CPU...")
    final_masks = Dict{String, Any}()
    
    body_mask = get(anatomy_masks, "body", nothing)
    body_gpu = nothing
    if body_mask !== nothing
        body_gpu = adapt(backend, map(x -> x > 0 ? UInt8(1) : UInt8(0), body_mask))
    end
    
    for (name, (ch, id)) in packed_tensor.registry
        m_cu = unpack_mask(packed_tensor, name)
        if any(m_cu .> 0)
            if body_gpu !== nothing
                m_cu .= m_cu .& body_gpu
            end
            cpu_mask = BitArray(adapt(Array, m_cu) .> 0)
            if any(cpu_mask)
                final_masks[name] = cpu_mask
            end
        end
    end
    
    return final_masks
end
