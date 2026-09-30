module VMCompiler

using ..VMOpcodes
using ..MaskPacker

export compile_level_instructions, can_compile_directly

function resolve_directional_margins(margins_mm::Any, side::Union{String, Nothing}=nothing)
    resolved = Dict{String, Float32}(
        "left" => 0.0f0, "right" => 0.0f0,
        "anterior" => 0.0f0, "posterior" => 0.0f0,
        "superior" => 0.0f0, "inferior" => 0.0f0
    )
    if margins_mm isa Number
        for k in keys(resolved)
            resolved[k] = Float32(margins_mm)
        end
        return resolved
    elseif !(margins_mm isa AbstractDict)
        for k in keys(resolved)
            resolved[k] = 10.0f0
        end
        return resolved
    end
    
    is_left_side = side !== nothing && lowercase(side) == "left"
    
    for (key, val) in margins_mm
        if string(key) == "slice_wise" || !(val isa Number)
            continue
        end
        v = Float32(val)
        if v <= 0f0
            continue
        end
        k = lowercase(string(key))
        if k == "all"
            for axis in keys(resolved)
                resolved[axis] = v
            end
        elseif k == "lateral"
            resolved[is_left_side ? "left" : "right"] = v
        elseif k == "medial"
            resolved[is_left_side ? "right" : "left"] = v
        elseif haskey(resolved, k)
            resolved[k] = v
        end
    end
    return resolved
end

function resolve_registry_entry(packed_tensor::PackedTensor, name::String, aliases::Dict{String, String})
    if isempty(name)
        return nothing
    end
    # Direct lookup
    if haskey(packed_tensor.registry, name)
        return packed_tensor.registry[name]
    end
    # Alias lookup
    nl = lowercase(name)
    if haskey(aliases, nl)
        alias_target = aliases[nl]
        if haskey(packed_tensor.registry, alias_target)
            return packed_tensor.registry[alias_target]
        end
    end
    # Case-insensitive lookup
    for (k, v) in packed_tensor.registry
        if lowercase(k) == nl
            return v
        end
    end
    return nothing
end

function resolve_exclusion_constituents(packed_tensor::PackedTensor, name::String, aliases::Dict{String, String})
    nl = lowercase(name)
    if nl in ["pelvis", "pelvic_bones"]
        return ["hip_left", "hip_right", "sacrum"]
    elseif nl == "vertebrae"
        return ["fused_spine"]
    elseif nl == "rib_1"
        return ["rib_left_1", "rib_right_1"]
    end
    # Check bilateral pair
    left_name = "$(name)_left"
    right_name = "$(name)_right"
    r_l = resolve_registry_entry(packed_tensor, left_name, aliases)
    r_r = resolve_registry_entry(packed_tensor, right_name, aliases)
    if r_l !== nothing || r_r !== nothing
        res = String[]
        if r_l !== nothing; push!(res, left_name); end
        if r_r !== nothing; push!(res, right_name); end
        return res
    end
    return String[]
end


"""
    can_compile_directly(rule_def, packed_tensor, aliases, out_name) -> Bool
Returns true if rule can be compiled into MegaKernel instructions.
Supports: expansion, boolean, mask copy, split, z-limit, geometric constraint types.
LCC is handled post-megakernel via GPU CCL.
"""
function can_compile_directly(rule_def::AbstractDict, packed_tensor::PackedTensor, aliases::Dict{String, String}, out_name::String)
    rule_type = get(rule_def, "rule", get(rule_def, "type", ""))
    
    # --- Determine rule category ---
    is_expansion = rule_type in ["DistanceExpansion", "DilatedMask", "MarginAroundLandmark", "AnisotropicMargin", "Morphology", "GenericMorphologyDAG"] ||
                   haskey(rule_def, "margins_mm") || haskey(rule_def, "margins") || haskey(rule_def, "margin_mm")
    
    is_boolean = rule_type in ["BitwiseOr", "BitwiseAnd", "Union", "Combine", "BooleanOperation"]
    
    is_mask_copy = rule_type in ["Mask", "Copy", "Base"]
    
    is_geometric_constraint = rule_type == "GeometricConstraint"
    
    is_limit_z = rule_type == "LimitZByLandmark"
    
    is_split_mask = rule_type == "SplitMask"
    
    # Only these types can compile
    if !(is_expansion || is_boolean || is_mask_copy || is_geometric_constraint || is_limit_z)
        println("can_compile_directly failed at line 126 for rule: ", out_name)
        return false
    end
    
    # --- Check that inputs exist in packed_tensor ---
    if is_expansion
        base_raw = get(rule_def, "base_landmark", get(rule_def, "input", get(rule_def, "input_mask", get(rule_def, "mask_name", get(rule_def, "landmark", get(rule_def, "landmarks", get(rule_def, "input_landmarks", get(rule_def, "base", ""))))))))
        if base_raw isa Dict || isempty(string(base_raw))
        println("can_compile_directly failed at line 133 for rule: ", out_name)
            return false
        end
        
        side_eff = ""
        if occursin("_left", lowercase(out_name)) || endswith(lowercase(out_name), "_l")
            side_eff = "left"
        elseif occursin("_right", lowercase(out_name)) || endswith(lowercase(out_name), "_r")
            side_eff = "right"
        elseif haskey(rule_def, "side")
            side_eff = lowercase(string(rule_def["side"]))
        end
        
        base_list = base_raw isa Vector ? base_raw : [base_raw]
        found_base = false
        for base_item in base_list
            base_str = string(base_item)
            isempty(base_str) && continue
            base_lookup = !isempty(side_eff) ? "$(base_str)_$side_eff" : base_str
            reg = resolve_registry_entry(packed_tensor, base_lookup, aliases)
            if reg === nothing
                reg = resolve_registry_entry(packed_tensor, base_str, aliases)
            end
            if reg !== nothing && reg[2] > 0
                bbox = get(packed_tensor.bboxes, base_lookup, get(packed_tensor.bboxes, base_str, nothing))
                if bbox !== nothing
                    found_base = true
                    break
                end
            end
        end
        if !found_base
        println("can_compile_directly failed at line 164 for rule: ", out_name)
            return false
        end
    end
    
    if is_boolean || is_mask_copy || is_geometric_constraint || is_limit_z
        # Check that at least one input is in packed_tensor
        # Boolean rules use various field names for their inputs
        inputs = get(rule_def, "components", get(rule_def, "inputs", get(rule_def, "targets", get(rule_def, "target_landmarks", get(rule_def, "landmarks", get(rule_def, "input", get(rule_def, "input_mask", get(rule_def, "mask_name", get(rule_def, "base_landmark", []))))))))  )
        if inputs isa String
            inputs = [inputs]
        end
        if !(inputs isa Vector) || isempty(inputs)
        println("can_compile_directly failed at line 176 for rule: ", out_name)
            return false
        end
        
        side_eff = ""
        if occursin("_left", lowercase(out_name)) || endswith(lowercase(out_name), "_l")
            side_eff = "left"
        elseif occursin("_right", lowercase(out_name)) || endswith(lowercase(out_name), "_r")
            side_eff = "right"
        elseif haskey(rule_def, "side")
            side_eff = lowercase(string(rule_def["side"]))
        end
        
        any_found = false
        for inp in inputs
            inp_str = string(inp)
            inp_lookup = !isempty(side_eff) ? "$(inp_str)_$side_eff" : inp_str
            reg = resolve_registry_entry(packed_tensor, inp_lookup, aliases)
            if reg === nothing
                reg = resolve_registry_entry(packed_tensor, inp_str, aliases)
            end
            if reg !== nothing
                any_found = true
                break
            end
        end
        if !any_found
        println("can_compile_directly failed at line 202 for rule: ", out_name)
            return false
        end
    end
    
    # --- Check constraints are compilable ---
    if haskey(rule_def, "constraints")
        for c in rule_def["constraints"]
            c isa Dict || continue
            c_type = get(c, "constraint_type", get(c, "type", ""))
            if c_type == "PlaneLimit"
                # Already supported via OP_PLANE
            elseif c_type in ["SuperiorTo", "InferiorTo", "AnteriorTo", "PosteriorTo", "RightOf", "LeftOf", "MedialTo", "LateralTo"]
                # These constraints are applied in postproc by RuleExecutors.apply_constraints
                # which has per-axis slice-wise bounds computation and proper offset handling.
                # Compiling them to OP_BBOX/OP_Z_BOUNDS produces different (incorrect) bounds.
                println("can_compile_directly failed at line 217 for rule: ", out_name)
                return false
            elseif c_type in ["SideLimit", "BetweenLandmarksXLimit", "SuperiorToLowestOf", "LineLimitBetweenLandmarks", "Superior", "Inferior"]
                # These are too complex for megakernel — fall back to interpreted
        println("can_compile_directly failed at line 220 for rule: ", out_name)
                return false
            else
        println("can_compile_directly failed at line 222 for rule: ", out_name)
                return false
            end
        end
    end
    
    # Check for layer_wise_propagation or dilation_ref (too complex)
    if haskey(rule_def, "layer_wise_propagation") || haskey(rule_def, "dilation_ref")
        println("can_compile_directly failed at line 229 for rule: ", out_name)
        return false
    end
    
    # Special postproc fields that megakernel can't handle
    if haskey(rule_def, "intersect_fat") || haskey(rule_def, "x_range_clip_mm") || haskey(rule_def, "x_range_clip")
        println("can_compile_directly failed at line 234 for rule: ", out_name)
        return false
    end
    if haskey(rule_def, "per_slice_lcc") && get(rule_def, "per_slice_lcc", false)
        println("can_compile_directly failed at line 237 for rule: ", out_name)
        return false  # Per-slice LCC requires slice-by-slice CPU processing
    end
    if haskey(rule_def, "dilate") || haskey(rule_def, "dilation")
        println("can_compile_directly failed at line 240 for rule: ", out_name)
        return false  # Post-LCC dilation not expressible in megakernel
    end
    if haskey(rule_def, "limit_landmark")
        println("can_compile_directly failed at line 243 for rule: ", out_name)
        return false  # Dynamic landmark-based Z limiting
    end
    
    # z_plane_restriction with superior/inferior fields - fall back to interpreted
    # The postproc apply_z_plane_restriction has proper physical-to-voxel conversion
    if haskey(rule_def, "z_plane_restriction")
        z_restr = rule_def["z_plane_restriction"]
        if z_restr isa Dict
            if haskey(z_restr, "superior") || haskey(z_restr, "inferior")
                println("can_compile_directly failed at line 259 for rule: ", out_name)
                return false
            end
        end
    end
    
    return true
end


"""
    compile_level_instructions(...) -> Vector{VMInstruction}
"""
function compile_level_instructions(
    output_names::Vector{String},
    rules::Dict{String, Dict},
    packed_tensor::PackedTensor,
    spacing::Tuple{Float64, Float64, Float64},
    dims::Tuple{Int, Int, Int},
    computed_landmarks::Dict{String, Any},
    origin::Tuple{Float64, Float64, Float64},
    direction::Tuple,
    aliases::Dict{String, String}
)
    instructions = VMInstruction[]
    sp_x = Float32(spacing[1]); sp_y = Float32(spacing[2]); sp_z = Float32(spacing[3])
    
    for (r, out_name) in enumerate(output_names)
        rule_idx = Int32(r)
        
        # Determine base rule definition and side
        base_rule_name = out_name
        side_eff = ""
        if !haskey(rules, base_rule_name)
            for suffix in ["_Left", "_Right", "_left", "_right"]
                if endswith(base_rule_name, suffix)
                    cand = base_rule_name[1:end-length(suffix)]
                    if haskey(rules, cand)
                        base_rule_name = cand
                        side_eff = lowercase(suffix[2:end])
                        break
                    end
                end
            end
        end
        rule_def = get(rules, base_rule_name, Dict())
        if isempty(side_eff)
            if occursin("_left", lowercase(out_name)) || endswith(lowercase(out_name), "_l")
                side_eff = "left"
            elseif occursin("_right", lowercase(out_name)) || endswith(lowercase(out_name), "_r")
                side_eff = "right"
            elseif haskey(rule_def, "side")
                side_eff = lowercase(string(rule_def["side"]))
            end
        end
        
        rule_type = get(rule_def, "rule", get(rule_def, "type", ""))
        
        # ====================================================================
        # Determine if this rule compiles directly
        # ====================================================================
        compiled_directly = false
        
        if can_compile_directly(rule_def, packed_tensor, aliases, out_name)
            
            # --- EXPANSION rules ---
            is_expansion = rule_type in ["DistanceExpansion", "DilatedMask", "MarginAroundLandmark", "AnisotropicMargin", "Morphology", "GenericMorphologyDAG"] ||
                           haskey(rule_def, "margins_mm") || haskey(rule_def, "margins") || haskey(rule_def, "margin_mm")
            
            if is_expansion
                margins = Dict{String, Any}()
                if rule_type in ["DistanceExpansion", "DilatedMask"]
                    d = Float64(get(rule_def, "distance_mm", get(rule_def, "dilation_mm", 20.0)))
                    margins["all"] = d
                elseif rule_type in ["Morphology", "GenericMorphologyDAG"]
                    r_val = Float64(get(rule_def, "radius_mm", 20.0))
                    op = lowercase(get(rule_def, "operation", "dilate"))
                    margins["all"] = (op == "erode" || op == "erosion") ? -r_val : r_val
                elseif rule_type == "MarginAroundLandmark"
                    m = Float64(get(rule_def, "margin_mm", get(rule_def, "margin", 10.0)))
                    margins["all"] = m
                elseif haskey(rule_def, "margins_mm")
                    m_raw = rule_def["margins_mm"]
                    margins = m_raw isa Dict ? m_raw : Dict("all" => Float64(m_raw))
                elseif haskey(rule_def, "margins")
                    m_raw = rule_def["margins"]
                    margins = m_raw isa Dict ? m_raw : Dict("all" => Float64(m_raw))
                elseif haskey(rule_def, "margin_mm")
                    margins = Dict("all" => Float64(rule_def["margin_mm"]))
                end
                
                base_raw = get(rule_def, "base_landmark", get(rule_def, "input", get(rule_def, "input_mask", get(rule_def, "mask_name", get(rule_def, "landmark", get(rule_def, "landmarks", get(rule_def, "input_landmarks", get(rule_def, "base", ""))))))))
                base_list = base_raw isa Vector ? base_raw : [base_raw]
                
                for base_item in base_list
                    base_str = string(base_item)
                    isempty(base_str) && continue
                    base_lookup = !isempty(side_eff) ? "$(base_str)_$side_eff" : base_str
                    reg = resolve_registry_entry(packed_tensor, base_lookup, aliases)
                    if reg === nothing
                        reg = resolve_registry_entry(packed_tensor, base_str, aliases)
                        if reg !== nothing
                            base_lookup = base_str
                        end
                    end
                    
                    if reg !== nothing && reg[2] > 0
                        ch, id = reg
                        bbox = get(packed_tensor.bboxes, base_lookup, nothing)
                        if bbox === nothing
                            bbox = get(packed_tensor.bboxes, base_str, nothing)
                        end
                        
                        if bbox !== nothing
                            min_i, max_i, min_j, max_j, min_k, max_k = bbox
                            res_m = resolve_directional_margins(margins, side_eff)
                            mx_pos = Float32(res_m["left"])
                            mx_neg = Float32(res_m["right"])
                            my_pos = Float32(res_m["posterior"])
                            my_neg = Float32(res_m["anterior"])
                            mz_pos = Float32(res_m["superior"])
                            mz_neg = Float32(res_m["inferior"])
                            
                            is_slice_wise = margins isa AbstractDict && get(margins, "slice_wise", false)
                            rad_x_pos = ceil(Int32, mx_pos / max(0.001f0, sp_x))
                            rad_x_neg = ceil(Int32, mx_neg / max(0.001f0, sp_x))
                            rad_y_pos = ceil(Int32, my_pos / max(0.001f0, sp_y))
                            rad_y_neg = ceil(Int32, my_neg / max(0.001f0, sp_y))
                            rad_z_pos = is_slice_wise ? Int32(0) : ceil(Int32, mz_pos / max(0.001f0, sp_z))
                            rad_z_neg = is_slice_wise ? Int32(0) : ceil(Int32, mz_neg / max(0.001f0, sp_z))
                            
                            push!(instructions, VMInstruction(
                                OP_ANISOTROPIC_EXPAND, rule_idx;
                                arg1=ch, arg2=id,
                                arg3=Int32(min_i), arg4=Int32(max_i),
                                arg5=Int32(min_j), arg6=Int32(max_j),
                                arg7=Int32(min_k), arg8=Int32(max_k),
                                arg9=rad_x_pos, arg10=rad_x_neg,
                                arg11=rad_y_pos, arg12=rad_y_neg,
                                arg13=rad_z_pos, arg14=rad_z_neg,
                                farg1=mx_pos, farg2=mx_neg,
                                farg3=my_pos, farg4=my_neg,
                                farg5=mz_pos, farg6=mz_neg
                            ))
                            
                            # DistanceExpansion excludes the base mask itself
                            if rule_type == "DistanceExpansion" || (rule_type == "MarginAroundLandmark" && !get(rule_def, "include_landmark", false))
                                push!(instructions, VMInstruction(OP_EXCLUDE_MASK, rule_idx; arg1=ch, arg2=id))
                            end
                            
                            compiled_directly = true
                        end
                    end
                end
            
            # --- BOOLEAN rules (BitwiseOr, Union, Combine) ---
            elseif rule_type in ["BitwiseOr", "Union", "Combine"]
                inputs = get(rule_def, "components", get(rule_def, "inputs", get(rule_def, "targets", get(rule_def, "target_landmarks", get(rule_def, "landmarks", get(rule_def, "input_landmarks", get(rule_def, "input", get(rule_def, "input_mask", get(rule_def, "mask_name", get(rule_def, "base_landmark", []))))))))))
                if inputs isa String; inputs = [inputs]; end
                method = lowercase(get(rule_def, "method", get(rule_def, "operation", get(rule_def, "op", "union"))))
                is_intersect = method in ["intersection", "intersect", "and"]
                first_loaded = false
                for inp in inputs
                    inp_str = string(inp)
                    inp_lookup = !isempty(side_eff) ? "$(inp_str)_$side_eff" : inp_str
                    reg = resolve_registry_entry(packed_tensor, inp_lookup, aliases)
                    if reg === nothing
                        reg = resolve_registry_entry(packed_tensor, inp_str, aliases)
                    end
                    if reg !== nothing
                        ch, id = reg
                        if !first_loaded
                            push!(instructions, VMInstruction(OP_LOAD_BASE_MASK, rule_idx; arg1=ch, arg2=id))
                            first_loaded = true
                        else
                            if is_intersect
                                push!(instructions, VMInstruction(OP_INTERSECT_MASK, rule_idx; arg1=ch, arg2=id))
                            else
                                push!(instructions, VMInstruction(OP_UNION_BASE_MASK, rule_idx; arg1=ch, arg2=id))
                            end
                        end
                    elseif is_intersect
                        # If any input to AND/intersection is missing, result is empty
                        empty!(instructions)
                        first_loaded = false
                        break
                    end
                end
                compiled_directly = first_loaded
            
            # --- BOOLEAN AND / GeometricConstraint (BitwiseAnd, BooleanOperation) ---
            elseif rule_type in ["BitwiseAnd", "BooleanOperation", "GeometricConstraint"]
                inputs = get(rule_def, "components", get(rule_def, "inputs", get(rule_def, "targets", get(rule_def, "target_landmarks", get(rule_def, "landmarks", get(rule_def, "input_landmarks", get(rule_def, "input", get(rule_def, "input_mask", get(rule_def, "mask_name", get(rule_def, "base_landmark", []))))))))))
                if inputs isa String; inputs = [inputs]; end
                operation = lowercase(get(rule_def, "operation", get(rule_def, "op", "and")))
                
                if operation in ["and", "intersect", "intersection"] || rule_type == "GeometricConstraint"
                    # Load first mask, then OP_INTERSECT_MASK for each additional
                    first_loaded = false
                    for (idx, inp) in enumerate(inputs)
                        inp_str = string(inp)
                        inp_lookup = !isempty(side_eff) ? "$(inp_str)_$side_eff" : inp_str
                        reg = resolve_registry_entry(packed_tensor, inp_lookup, aliases)
                        if reg === nothing
                            reg = resolve_registry_entry(packed_tensor, inp_str, aliases)
                        end
                        if reg !== nothing
                            ch, id = reg
                            if !first_loaded
                                push!(instructions, VMInstruction(OP_LOAD_BASE_MASK, rule_idx; arg1=ch, arg2=id))
                                first_loaded = true
                            else
                                push!(instructions, VMInstruction(OP_INTERSECT_MASK, rule_idx; arg1=ch, arg2=id))
                            end
                        else
                            # If any input to AND is missing, the result is empty
                            empty!(instructions)
                            first_loaded = false
                            break
                        end
                    end
                    compiled_directly = first_loaded
                elseif operation in ["or", "union"]
                    first_loaded = false
                    for inp in inputs
                        inp_str = string(inp)
                        inp_lookup = !isempty(side_eff) ? "$(inp_str)_$side_eff" : inp_str
                        reg = resolve_registry_entry(packed_tensor, inp_lookup, aliases)
                        if reg === nothing
                            reg = resolve_registry_entry(packed_tensor, inp_str, aliases)
                        end
                        if reg !== nothing
                            ch, id = reg
                            if !first_loaded
                                push!(instructions, VMInstruction(OP_LOAD_BASE_MASK, rule_idx; arg1=ch, arg2=id))
                                first_loaded = true
                            else
                                push!(instructions, VMInstruction(OP_UNION_BASE_MASK, rule_idx; arg1=ch, arg2=id))
                            end
                        end
                    end
                    compiled_directly = first_loaded
                end
            
            # --- MASK COPY (Mask, Copy, Base) ---
            elseif rule_type in ["Mask", "Copy", "Base"]
                inputs = get(rule_def, "inputs", get(rule_def, "targets", get(rule_def, "target_landmarks", get(rule_def, "landmarks", get(rule_def, "input", get(rule_def, "input_mask", get(rule_def, "mask_name", get(rule_def, "base_landmark", [])))))))  )
                if inputs isa String; inputs = [inputs]; end
                first_loaded = false
                for inp in inputs
                    inp_str = string(inp)
                    inp_lookup = !isempty(side_eff) ? "$(inp_str)_$side_eff" : inp_str
                    reg = resolve_registry_entry(packed_tensor, inp_lookup, aliases)
                    if reg === nothing
                        reg = resolve_registry_entry(packed_tensor, inp_str, aliases)
                    end
                    if reg !== nothing
                        ch, id = reg
                        if !first_loaded
                            push!(instructions, VMInstruction(OP_LOAD_BASE_MASK, rule_idx; arg1=ch, arg2=id))
                            first_loaded = true
                        else
                            push!(instructions, VMInstruction(OP_UNION_BASE_MASK, rule_idx; arg1=ch, arg2=id))
                        end
                    end
                end
                compiled_directly = first_loaded
            
            # --- SPLIT MASK ---
            elseif rule_type == "SplitMask"
                # SplitMask: load input, then restrict to one half via OP_BBOX
                input_name = string(get(rule_def, "input", get(rule_def, "input_mask", get(rule_def, "mask_name", ""))))
                inp_lookup = !isempty(side_eff) ? "$(input_name)_$side_eff" : input_name
                reg = resolve_registry_entry(packed_tensor, inp_lookup, aliases)
                if reg === nothing
                    reg = resolve_registry_entry(packed_tensor, input_name, aliases)
                end
                if reg !== nothing
                    ch, id = reg
                    push!(instructions, VMInstruction(OP_LOAD_BASE_MASK, rule_idx; arg1=ch, arg2=id))
                    
                    # Determine split axis and threshold
                    axis_str = lowercase(get(rule_def, "axis", "z"))
                    axis_code = axis_str == "x" ? Int32(1) : (axis_str == "y" ? Int32(2) : Int32(3))
                    keep_str = lowercase(get(rule_def, "keep", "min"))
                    
                    # Get bounding box for the input to compute split point
                    inp_bbox_key = !isempty(side_eff) ? "$(input_name)_$side_eff" : input_name
                    bbox = get(packed_tensor.bboxes, inp_bbox_key, get(packed_tensor.bboxes, input_name, nothing))
                    if bbox !== nothing
                        min_i, max_i, min_j, max_j, min_k, max_k = bbox
                        
                        split_method = lowercase(get(rule_def, "split_method", "centroid"))
                        split_val = Int32(-1)  # sentinel: -1 means "can't compute"
                        
                        if split_method == "centroid"
                            # True centroid requires GPU reduction — can't compile in megakernel
                            # Leave split_val = -1 to skip compilation
                        elseif split_method == "center"
                            # Bounding box center
                            if axis_code == Int32(1)
                                split_val = Int32(div(min_i + max_i, 2))
                            elseif axis_code == Int32(2)
                                split_val = Int32(div(min_j + max_j, 2))
                            else
                                split_val = Int32(div(min_k + max_k, 2))
                            end
                        elseif split_method == "image_center"
                            if axis_code == Int32(1)
                                split_val = Int32(div(dims[1], 2))
                            elseif axis_code == Int32(2)
                                split_val = Int32(div(dims[2], 2))
                            else
                                split_val = Int32(div(dims[3], 2))
                            end
                        elseif startswith(split_method, "ratio_")
                            ratio = parse(Float32, replace(split_method, "ratio_" => ""))
                            if axis_code == Int32(1)
                                split_val = Int32(floor(Float32(min_i - 1) + Float32(max_i - min_i) * ratio)) + Int32(1)
                            elseif axis_code == Int32(2)
                                split_val = Int32(floor(Float32(min_j - 1) + Float32(max_j - min_j) * ratio)) + Int32(1)
                            else
                                split_val = Int32(floor(Float32(min_k - 1) + Float32(max_k - min_k) * ratio)) + Int32(1)
                            end
                        else
                            split_val = Int32(div(min_k + max_k, 2))
                        end
                        
                        if split_val > Int32(0)
                            if keep_str == "min"
                                push!(instructions, VMInstruction(OP_BBOX, rule_idx; arg1=axis_code, arg2=Int32(1), arg3=split_val))
                            else  # "max"
                                push!(instructions, VMInstruction(OP_BBOX, rule_idx; arg1=axis_code, arg2=split_val, arg3=Int32(axis_code == 1 ? dims[1] : (axis_code == 2 ? dims[2] : dims[3]))))
                            end
                            compiled_directly = true
                        end
                    end
                end
            
            # --- LIMIT Z BY LANDMARK ---
            elseif rule_type == "LimitZByLandmark"
                input_name = string(get(rule_def, "input", get(rule_def, "input_mask", get(rule_def, "mask_name", ""))))
                inp_lookup = !isempty(side_eff) ? "$(input_name)_$side_eff" : input_name
                reg = resolve_registry_entry(packed_tensor, inp_lookup, aliases)
                if reg === nothing
                    reg = resolve_registry_entry(packed_tensor, input_name, aliases)
                end
                if reg !== nothing
                    ch, id = reg
                    push!(instructions, VMInstruction(OP_LOAD_BASE_MASK, rule_idx; arg1=ch, arg2=id))
                    
                    # Determine Z bounds from landmarks
                    z_min_lm = get(rule_def, "z_min_landmark", nothing)
                    z_max_lm = get(rule_def, "z_max_landmark", nothing)
                    k_min = Int32(1)
                    k_max = Int32(dims[3])
                    
                    if z_min_lm !== nothing
                        z_min_str = string(z_min_lm)
                        if !isempty(side_eff); z_min_str_s = "$(z_min_str)_$side_eff"; else; z_min_str_s = z_min_str; end
                        for try_key in [z_min_str_s, z_min_str]
                            if haskey(computed_landmarks, try_key)
                                val = computed_landmarks[try_key]
                                if val isa Number
                                    k_min = Int32(max(1, round(Int, val)))
                                elseif val isa Tuple || val isa Vector
                                    if length(val) >= 3
                                        k_min = Int32(max(1, round(Int, val[3])))
                                    end
                                end
                                break
                            end
                        end
                    end
                    
                    if z_max_lm !== nothing
                        z_max_str = string(z_max_lm)
                        if !isempty(side_eff); z_max_str_s = "$(z_max_str)_$side_eff"; else; z_max_str_s = z_max_str; end
                        for try_key in [z_max_str_s, z_max_str]
                            if haskey(computed_landmarks, try_key)
                                val = computed_landmarks[try_key]
                                if val isa Number
                                    k_max = Int32(min(dims[3], round(Int, val)))
                                elseif val isa Tuple || val isa Vector
                                    if length(val) >= 3
                                        k_max = Int32(min(dims[3], round(Int, val[3])))
                                    end
                                end
                                break
                            end
                        end
                    end
                    
                    push!(instructions, VMInstruction(OP_Z_BOUNDS, rule_idx; arg1=k_min, arg2=k_max))
                    compiled_directly = true
                end
            end  # end rule type dispatch
        end  # end can_compile_directly check
        

        # ====================================================================
        # Compile constraints (for directly compiled rules)
        # ====================================================================
        if compiled_directly
            for c in get(rule_def, "constraints", [])
                c isa Dict || continue
                c_type = get(c, "constraint_type", get(c, "type", ""))
                
                if c_type == "PlaneLimit"
                    lm_key = get(c, "landmark", "")
                    plane_def = nothing
                    keys_to_try = [lm_key]
                    if side_eff !== nothing && !isempty(side_eff)
                        insert!(keys_to_try, 1, "$(lm_key)_$(lowercase(side_eff))")
                    end
                    for k in keys_to_try
                        if haskey(computed_landmarks, k)
                            val = computed_landmarks[k]
                            if val isa Tuple || val isa Vector
                                plane_def = val
                                break
                            elseif val isa Number
                                plane_def = [Float64(origin[1] + dims[1]*spacing[1]/2.0), Float64(origin[2] + dims[2]*spacing[2]/2.0), Float64(val)]
                                break
                            end
                        end
                    end
                    
                    ref_pt = nothing
                    normal = get(c, "normal", nothing)
                    if plane_def !== nothing && length(plane_def) == 2
                        ref_pt = plane_def[1]
                        normal = plane_def[2]
                    elseif plane_def !== nothing && length(plane_def) >= 3
                        ref_pt = [Float64(plane_def[1]), Float64(plane_def[2]), Float64(plane_def[3])]
                        if normal === nothing
                            normal = [0.0, 0.0, 1.0]
                        end
                    end
                    
                    if normal === nothing
                        normal = [0.0, 1.0, 0.0]
                    end
                    
                    if ref_pt !== nothing && length(ref_pt) >= 3 && length(normal) >= 3
                        off_mm = Float32(get(c, "offset_mm", 0.0))
                        norm_vec = [Float32(normal[1]), Float32(normal[2]), Float32(normal[3])]
                        n_len = sqrt(sum(norm_vec .^ 2))
                        if n_len > 0.0f0
                            norm_vec ./= n_len
                            pt_vec = [Float32(ref_pt[1]), Float32(ref_pt[2]), Float32(ref_pt[3])] + off_mm .* norm_vec
                            
                            keep_neg = (get(c, "side_to_keep", "negative") == "negative")
                            
                            dir_00 = Float32(direction[1])
                            dir_11 = Float32(direction[5])
                            dir_22 = Float32(direction[9])
                            sp_x_l = Float32(spacing[1])
                            sp_y_l = Float32(spacing[2])
                            sp_z_l = Float32(spacing[3])
                            orig_x = Float32(origin[1]) - sp_x_l * dir_00
                            orig_y = Float32(origin[2]) - sp_y_l * dir_11
                            orig_z = Float32(origin[3]) - sp_z_l * dir_22
                            
                            nx_eff = sp_x_l * dir_00 * norm_vec[1]
                            ny_eff = sp_y_l * dir_11 * norm_vec[2]
                            nz_eff = sp_z_l * dir_22 * norm_vec[3]
                            d_eff = (orig_x - pt_vec[1]) * norm_vec[1] + (orig_y - pt_vec[2]) * norm_vec[2] + (orig_z - pt_vec[3]) * norm_vec[3]
                            
                            push!(instructions, VMInstruction(OP_PLANE, rule_idx; arg1 = keep_neg ? -1 : 1, farg1=nx_eff, farg2=ny_eff, farg3=nz_eff, farg4=d_eff))
                        end
                    end
                
                elseif c_type in ["SuperiorTo", "InferiorTo", "AnteriorTo", "PosteriorTo", "RightOf", "LeftOf", "MedialTo", "LateralTo"]
                    # Emit OP_Z_BOUNDS or OP_BBOX using resolved bounding box
                    # Only non-slice-wise constraints reach here (slice-wise rejected in can_compile_directly)
                    lm_name = string(get(c, "landmark", get(c, "mask_name", "")))
                    lm_lookup = !isempty(side_eff) ? "$(lm_name)_$side_eff" : lm_name
                    bbox = get(packed_tensor.bboxes, lm_lookup, get(packed_tensor.bboxes, lm_name, nothing))
                    
                    if bbox !== nothing
                        # bbox = (min_i, max_i, min_j, max_j, min_k, max_k)
                        offset_mm = Float64(get(c, "offset_mm", 0.0))
                        b_part = string(get(c, "boundary_part", "min"))
                        
                        # Resolve MedialTo/LateralTo to LeftOf/RightOf
                        actual_type = c_type
                        if c_type == "MedialTo"
                            actual_type = (side_eff == "left") ? "RightOf" : "LeftOf"
                        elseif c_type == "LateralTo"
                            actual_type = (side_eff == "left") ? "LeftOf" : "RightOf"
                        end
                        
                        if actual_type == "SuperiorTo"
                            # Must be above (superior to) landmark: keep voxels with k >= bound
                            bound_k = b_part == "max" ? bbox[6] : (b_part == "center" ? div(bbox[5] + bbox[6], 2) : bbox[5])
                            offset_slices = round(Int, offset_mm / spacing[3])
                            k_lo = Int32(clamp(bound_k + offset_slices, 1, dims[3]))
                            push!(instructions, VMInstruction(OP_Z_BOUNDS, rule_idx; arg1=k_lo, arg2=Int32(dims[3])))
                        elseif actual_type == "InferiorTo"
                            # Must be below (inferior to) landmark: keep voxels with k <= bound
                            bound_k = b_part == "max" ? bbox[6] : (b_part == "center" ? div(bbox[5] + bbox[6], 2) : bbox[5])
                            offset_slices = round(Int, offset_mm / spacing[3])
                            k_hi = Int32(clamp(bound_k + offset_slices, 1, dims[3]))
                            push!(instructions, VMInstruction(OP_Z_BOUNDS, rule_idx; arg1=Int32(1), arg2=k_hi))
                        elseif actual_type == "AnteriorTo"
                            # Must be anterior (lower Y in RAS) to landmark: keep voxels with j <= bound
                            bound_j = b_part == "max" ? bbox[4] : (b_part == "center" ? div(bbox[3] + bbox[4], 2) : bbox[3])
                            offset_vox = round(Int, offset_mm / spacing[2])
                            j_hi = Int32(clamp(bound_j + offset_vox, 1, dims[2]))
                            push!(instructions, VMInstruction(OP_BBOX, rule_idx; arg1=Int32(2), arg2=Int32(1), arg3=j_hi))
                        elseif actual_type == "PosteriorTo"
                            # Must be posterior (higher Y in RAS) to landmark: keep voxels with j >= bound
                            bound_j = b_part == "max" ? bbox[4] : (b_part == "center" ? div(bbox[3] + bbox[4], 2) : bbox[3])
                            offset_vox = round(Int, offset_mm / spacing[2])
                            j_lo = Int32(clamp(bound_j + offset_vox, 1, dims[2]))
                            push!(instructions, VMInstruction(OP_BBOX, rule_idx; arg1=Int32(2), arg2=j_lo, arg3=Int32(dims[2])))
                        elseif actual_type == "RightOf"
                            # Keep voxels with i <= bound (right side in patient coords = lower X index)
                            bound_i = b_part == "max" ? bbox[2] : (b_part == "center" ? div(bbox[1] + bbox[2], 2) : bbox[1])
                            offset_vox = round(Int, offset_mm / spacing[1])
                            i_hi = Int32(clamp(bound_i + offset_vox, 1, dims[1]))
                            push!(instructions, VMInstruction(OP_BBOX, rule_idx; arg1=Int32(1), arg2=Int32(1), arg3=i_hi))
                        elseif actual_type == "LeftOf"
                            # Keep voxels with i >= bound (left side in patient coords = higher X index)
                            bound_i = b_part == "max" ? bbox[2] : (b_part == "center" ? div(bbox[1] + bbox[2], 2) : bbox[1])
                            offset_vox = round(Int, offset_mm / spacing[1])
                            i_lo = Int32(clamp(bound_i + offset_vox, 1, dims[1]))
                            push!(instructions, VMInstruction(OP_BBOX, rule_idx; arg1=Int32(1), arg2=i_lo, arg3=Int32(dims[1])))
                        end
                    end
                end
            end
            
            # --- Z plane restriction ---
            if haskey(rule_def, "z_plane_restriction")
                z_restr = rule_def["z_plane_restriction"]
                if z_restr isa AbstractDict
                    z_min_key = get(z_restr, "z_min_landmark", nothing)
                    z_max_key = get(z_restr, "z_max_landmark", nothing)
                    k_min_r = Int32(1)
                    k_max_r = Int32(dims[3])
                    
                    # Legacy format: z_min_landmark / z_max_landmark
                    for (z_key, is_min) in [(z_min_key, true), (z_max_key, false)]
                        z_key === nothing && continue
                        z_str = string(z_key)
                        for try_k in [!isempty(side_eff) ? "$(z_str)_$side_eff" : z_str, z_str]
                            if haskey(computed_landmarks, try_k)
                                val = computed_landmarks[try_k]
                                z_val = val isa Number ? round(Int, val) : (val isa Tuple || val isa Vector) && length(val) >= 3 ? round(Int, val[3]) : nothing
                                if z_val !== nothing
                                    if is_min
                                        k_min_r = Int32(max(1, z_val))
                                    else
                                        k_max_r = Int32(min(dims[3], z_val))
                                    end
                                end
                                break
                            end
                        end
                    end
                    
                    # New format: superior / inferior (resolve from packed_tensor.bboxes)
                    sup_spec = get(z_restr, "superior", nothing)
                    inf_spec = get(z_restr, "inferior", nothing)
                    
                    if sup_spec !== nothing
                        sup_lm = string(sup_spec)
                        sup_part = string(get(z_restr, "superior_part", "max"))
                        sup_off = Float64(get(z_restr, "superior_offset_mm", 0.0))
                        sup_lm_lookup = !isempty(side_eff) ? "$(sup_lm)_$side_eff" : sup_lm
                        sup_bbox = get(packed_tensor.bboxes, sup_lm_lookup, get(packed_tensor.bboxes, sup_lm, nothing))
                        if sup_bbox !== nothing
                            sup_k = sup_part == "absolute_max" ? sup_bbox[6] : (sup_part == "max" ? sup_bbox[6] : (sup_part == "center" ? div(sup_bbox[5] + sup_bbox[6], 2) : sup_bbox[5]))
                            offset_slices = round(Int, sup_off / spacing[3])
                            k_max_r = Int32(clamp(sup_k + offset_slices, 1, dims[3]))
                        end
                    end
                    
                    if inf_spec !== nothing
                        inf_lm = string(inf_spec)
                        inf_part = string(get(z_restr, "inferior_part", "min"))
                        inf_off = Float64(get(z_restr, "inferior_offset_mm", 0.0))
                        inf_lm_lookup = !isempty(side_eff) ? "$(inf_lm)_$side_eff" : inf_lm
                        inf_bbox = get(packed_tensor.bboxes, inf_lm_lookup, get(packed_tensor.bboxes, inf_lm, nothing))
                        if inf_bbox !== nothing
                            inf_k = inf_part == "absolute_min" ? inf_bbox[5] : (inf_part == "min" ? inf_bbox[5] : (inf_part == "center" ? div(inf_bbox[5] + inf_bbox[6], 2) : inf_bbox[6]))
                            offset_slices = round(Int, inf_off / spacing[3])
                            k_min_r = Int32(clamp(inf_k + offset_slices, 1, dims[3]))
                        end
                    end
                    
                    push!(instructions, VMInstruction(OP_Z_BOUNDS, rule_idx; arg1=k_min_r, arg2=k_max_r))
                end
            end
        end

        # ====================================================================
        # Fallback: Load from pre-computed output buffer
        # ====================================================================
        if !compiled_directly
            push!(instructions, VMInstruction(OP_LOAD_OUTPUT_BUFFER, rule_idx))
        end
        
        # ====================================================================
        # Append landmarks (unioned with active) — for compiled rules
        # ====================================================================
        if compiled_directly
            for app_lm in get(rule_def, "append_landmarks", [])
                app_lookup = !isempty(side_eff) ? "$(app_lm)_$side_eff" : app_lm
                reg = resolve_registry_entry(packed_tensor, app_lookup, aliases)
                if reg === nothing
                    reg = resolve_registry_entry(packed_tensor, String(app_lm), aliases)
                end
                if reg !== nothing
                    ch, id = reg
                    push!(instructions, VMInstruction(OP_UNION_BASE_MASK, rule_idx; arg1=ch, arg2=id))
                end
            end
        end
        
        # ====================================================================
        # Exclusions — for ALL rules (fused into megakernel)
        # ====================================================================
        ex_list = unique(vcat(get(rule_def, "exclude", []), get(rule_def, "exclude_structures", []), get(rule_def, "exclude_landmarks", [])))
        for ex_raw in ex_list
            ex_name = ""
            ex_margin = 0.0
            if ex_raw isa Dict
                ex_name = string(get(ex_raw, "landmark", get(ex_raw, "name", "")))
                ex_margin = Float64(get(ex_raw, "margin_mm", 0.0))
            else
                ex_name = string(ex_raw)
            end
            isempty(ex_name) && continue
            
            ex_lookup = !isempty(side_eff) ? "$(ex_name)_$side_eff" : ex_name
            reg = resolve_registry_entry(packed_tensor, ex_lookup, aliases)
            if reg === nothing
                reg = resolve_registry_entry(packed_tensor, ex_name, aliases)
            end
            if reg !== nothing
                ch, id = reg
                if ex_margin == 0.0
                    push!(instructions, VMInstruction(OP_EXCLUDE_MASK, rule_idx; arg1=ch, arg2=id))
                elseif compiled_directly
                    rx = ceil(Int32, ex_margin / max(0.001f0, sp_x))
                    ry = ceil(Int32, ex_margin / max(0.001f0, sp_y))
                    rz = ceil(Int32, ex_margin / max(0.001f0, sp_z))
                    push!(instructions, VMInstruction(OP_EXCLUDE_MARGIN, rule_idx; arg1=ch, arg2=id, arg3=rx, arg4=ry, arg5=rz))
                end
            else
                constituents = resolve_exclusion_constituents(packed_tensor, ex_name, aliases)
                for c_name in constituents
                    c_reg = resolve_registry_entry(packed_tensor, c_name, aliases)
                    if c_reg !== nothing
                        ch, id = c_reg
                        if ex_margin == 0.0
                            push!(instructions, VMInstruction(OP_EXCLUDE_MASK, rule_idx; arg1=ch, arg2=id))
                        elseif compiled_directly
                            rx = ceil(Int32, ex_margin / max(0.001f0, sp_x))
                            ry = ceil(Int32, ex_margin / max(0.001f0, sp_y))
                            rz = ceil(Int32, ex_margin / max(0.001f0, sp_z))
                            push!(instructions, VMInstruction(OP_EXCLUDE_MARGIN, rule_idx; arg1=ch, arg2=id, arg3=rx, arg4=ry, arg5=rz))
                        end
                    end
                end
            end
        end
        
        # ====================================================================
        # Write output
        # ====================================================================
        push!(instructions, VMInstruction(OP_WRITE_OUTPUT, rule_idx))
    end
    
    return instructions
end

end # module
