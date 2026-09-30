"""
Module `StaticArena`

Provides statically allocated memory buffers (arenas) for GPU computations.
This prevents memory fragmentation and Out-Of-Memory (OOM) errors caused by dynamically
allocating large 4D tensors and intermediate masks during DAG execution and exact distance transforms (EDT).

It holds three main arenas:
1. `EDT_ARENA`: For exact distance transforms (ping-pong distance arrays and parabola vertex tracking).
2. `MASK_ARENA`: For temporary 3D bitmasks in generic rule executors.
3. `VM_ARENA`: For holding the massive 4D Step Output tensor of the DAG execution step.
"""
module StaticArena
using CUDA
using KernelAbstractions

# Global EDT buffers
const EDT_ARENA = Dict{Symbol, Any}()

# Global Mask buffers for general RuleExecutors processing
const MASK_ARENA = Dict{Int, Any}()
const MASK_IN_USE = Dict{Int, Bool}()

function init_edt_arena(backend, dims)
    dx, dy, dz = Int32(dims[1]), Int32(dims[2]), Int32(dims[3])
    max_dim = max(dx, dy, dz)
    
    n_x = Int(dy) * Int(dz)
    n_y = Int(dx) * Int(dz)
    n_z = Int(dx) * Int(dy)
    
    # Pre-allocate exactly the buffers needed for gpu_exact_edt
    EDT_ARENA[:D_a] = KernelAbstractions.allocate(backend, Float32, dims)
    EDT_ARENA[:D_b] = KernelAbstractions.allocate(backend, Float32, dims)
    
    EDT_ARENA[:v_x] = KernelAbstractions.allocate(backend, Int32, (Int(max_dim), n_x))
    EDT_ARENA[:z_x] = KernelAbstractions.allocate(backend, Float32, (Int(max_dim) + 1, n_x))
    
    EDT_ARENA[:v_y] = KernelAbstractions.allocate(backend, Int32, (Int(max_dim), n_y))
    EDT_ARENA[:z_y] = KernelAbstractions.allocate(backend, Float32, (Int(max_dim) + 1, n_y))
    
    EDT_ARENA[:v_z] = KernelAbstractions.allocate(backend, Int32, (Int(max_dim), n_z))
    EDT_ARENA[:z_z] = KernelAbstractions.allocate(backend, Float32, (Int(max_dim) + 1, n_z))
end

function init_mask_arena(backend, dims; num_masks=10)
    empty!(MASK_ARENA)
    empty!(MASK_IN_USE)
    for i in 1:num_masks
        MASK_ARENA[i] = KernelAbstractions.allocate(backend, UInt8, dims)
        MASK_IN_USE[i] = false
    end
end

function acquire_mask(backend)
    for (k, in_use) in MASK_IN_USE
        if !in_use
            MASK_IN_USE[k] = true
            # Fill with 0
            if backend isa CUDABackend
                CUDA.fill!(MASK_ARENA[k], UInt8(0))
            else
                fill!(MASK_ARENA[k], UInt8(0))
            end
            return MASK_ARENA[k], k
        end
    end
    error("MASK_ARENA exhausted! Increase num_masks in init_mask_arena.")
end

function release_mask(k::Int)
    if haskey(MASK_IN_USE, k)
        MASK_IN_USE[k] = false
    end
end

const VM_ARENA = Dict{Symbol, Any}()

function init_vm_arena(backend, dims::Tuple{Int, Int, Int}; max_rules=64)
    # level_output will store up to `max_rules` masks produced in a single level.
    # 512 x 512 x 284 x 25 UInt8 = ~1.74 GB
    # Max observed outputs per level is 21, so 25 gives ample headroom without OOMing when other julia processes are running.
    VM_ARENA[:level_output] = KernelAbstractions.zeros(backend, UInt8, dims[1], dims[2], dims[3], max_rules)
    println("  [StaticArena] VM level_output preallocated: $(dims[1])×$(dims[2])×$(dims[3])×$(max_rules) UInt8 = $(round(prod(dims) * max_rules / 1024^3, digits=2)) GiB")
end

function reset_level_output!(backend, num_rules::Int)
    if haskey(VM_ARENA, :level_output)
        out_buf = VM_ARENA[:level_output]
        cap = size(out_buf, 4)
        if num_rules > cap
            error("VM_ARENA[:level_output] capacity $cap exceeded by $num_rules rules! Increase max_rules in init_vm_arena.")
        end
        v = view(out_buf, :, :, :, 1:num_rules)
        if backend isa CUDABackend
            CUDA.fill!(v, UInt8(0))
        else
            fill!(v, UInt8(0))
        end
    end
end

# ============================================================
# CCL Arena: Preallocated buffers for GPU Connected Component Labeling
# ============================================================
const CCL_ARENA = Dict{Symbol, Any}()

function init_ccl_arena(backend, dims)
    n = dims[1] * dims[2] * dims[3]
    CCL_ARENA[:labels] = KernelAbstractions.allocate(backend, Int32, n)   # labels buffer (1D)
    CCL_ARENA[:counts] = KernelAbstractions.allocate(backend, Int32, n)   # counts buffer (1D)
    CCL_ARENA[:output] = KernelAbstractions.allocate(backend, UInt8, dims) # output mask buffer
    CCL_ARENA[:block_val] = KernelAbstractions.allocate(backend, Int32, 1024)
    CCL_ARENA[:block_idx] = KernelAbstractions.allocate(backend, Int32, 1024)
    println("  [StaticArena] CCL_ARENA preallocated: labels+counts+output+blocks = $(round((2*n*4 + prod(dims))/1024^3, digits=2)) GiB")
end

# ============================================================
# FT Arena: Preallocated buffers for GPU Feature Transform
# Used by anisotropic expansion for directional margin thresholding
# ============================================================
const FT_ARENA = Dict{Symbol, Any}()

function init_ft_arena(backend, dims)
    # Ping-pong buffers for feature coordinates during separable passes
    # Fx_a/Fx_b: X coordinate of nearest seed (Int16 sufficient for dims ≤ 32767)
    # Fy_b: Y coordinate (Fy is written by Y pass, read by Z pass — needs ping-pong)
    # Fz: Z coordinate (written by Z pass only, no ping-pong needed)
    FT_ARENA[:Fx_a] = KernelAbstractions.allocate(backend, Int16, dims)
    FT_ARENA[:Fx_b] = KernelAbstractions.allocate(backend, Int16, dims)
    FT_ARENA[:Fy]   = KernelAbstractions.allocate(backend, Int16, dims)  # init writes here
    FT_ARENA[:Fy_b] = KernelAbstractions.allocate(backend, Int16, dims)  # Z pass writes here
    FT_ARENA[:Fz]   = KernelAbstractions.allocate(backend, Int16, dims)
    total_bytes = 5 * prod(dims) * 2  # 5 buffers × Int16 (2 bytes)
    println("  [StaticArena] FT_ARENA preallocated: 5×Int16 = $(round(total_bytes/1024^3, digits=2)) GiB")
end

end # module
