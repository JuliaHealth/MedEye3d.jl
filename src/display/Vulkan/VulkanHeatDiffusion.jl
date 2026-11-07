"""
    VulkanHeatDiffusion

GPU compute module that runs the heat diffusion PDE entirely on the Vulkan GPU.
This enables sub-millisecond interactive segmentation by avoiding any CPU↔GPU roundtrips
during the user's click-and-hold interaction.

## Pipeline:
1. `init_heat_diffusion!` — Creates compute pipelines, allocates 3D storage images
2. `upload_diffusivity!` — Uploads the precomputed D(x) field to GPU
3. `run_heat_diffusion!` — Seeds + runs K diffusion steps + extracts mask
4. `read_mask_to_cpu` — Async reads the mask back to CPU for saving
5. `destroy_heat_diffusion!` — Cleans up all resources
"""
module VulkanHeatDiffusion

using Vulkan
using VulkanCore
using Logging
using ..VulkanShaders: compile_glsl_to_spirv

export HeatDiffusionState, init_heat_diffusion!, destroy_heat_diffusion!
export upload_diffusivity!, run_heat_diffusion!, read_mask_to_cpu
export seed_only!, run_incremental_steps!, compute_frontier_edge_scale, compute_local_relative_edge_params, compute_directional_relative_edge_params

# ─── Push Constants ──────────────────────────────────────────────────────

struct SeedInitPC
    seed_x::Int32
    seed_y::Int32
    seed_z::Int32
    seed_radius::Int32
    box_min_x::Int32
    box_min_y::Int32
    box_min_z::Int32
    box_min_w::Int32
    inv_dx2::Float32
    inv_dy2::Float32
    inv_dz2::Float32
    _pad::Float32
end

struct DiffuseStepPC
    dt::Float32
    use_box::Int32
    seed_x::Int32
    seed_y::Int32
    box_min_x::Int32
    box_min_y::Int32
    box_min_z::Int32
    box_min_w::Int32
    box_max_x::Int32
    box_max_y::Int32
    box_max_z::Int32
    box_max_w::Int32
    inv_dx2::Float32
    inv_dy2::Float32
    inv_dz2::Float32
    s_boost::Float32
    # Directional peaks (6 floats) & metadata (6 floats) -> 12 * 4 = 48 bytes -> total 112 bytes
    peak_xp::Float32
    peak_xm::Float32
    peak_yp::Float32
    peak_ym::Float32
    peak_zp::Float32
    peak_zm::Float32
    edge_bg::Float32
    edge_clip::Float32
    gamma_eff::Float32
    is_directional::Float32
    edge_barrier::Float32
    is_tubular::Float32
end

struct ExtractMaskPC
    theta::Float32
    tau::Float32
    hard_thresh::Int32
    use_box::Int32
    box_min_x::Int32
    box_min_y::Int32
    box_min_z::Int32
    box_min_w::Int32
    box_max_x::Int32
    box_max_y::Int32
    box_max_z::Int32
    box_max_w::Int32
end

# ─── State struct ────────────────────────────────────────────────────────

"""
    HeatDiffusionState

Holds all Vulkan resources for the heat diffusion compute pipeline.
"""
mutable struct HeatDiffusionState
    # Volume dimensions
    width::Int
    height::Int
    depth::Int
    
    # 3D storage images (VK_FORMAT_R32_SFLOAT)
    img_diffusivity::Image          # Precomputed D(x) field
    img_u_ping::Image               # Heat field buffer A
    img_u_pong::Image               # Heat field buffer B
    img_mask::Image                 # Output binary mask
    
    # Image views
    view_diffusivity::ImageView
    view_u_ping::ImageView
    view_u_pong::ImageView
    view_mask::ImageView
    
    # Device memory allocations
    mem_diffusivity::DeviceMemory
    mem_u_ping::DeviceMemory
    mem_u_pong::DeviceMemory
    mem_mask::DeviceMemory
    
    # Compute pipelines
    pipeline_seed_init::Pipeline
    pipeline_diffuse_step::Pipeline
    pipeline_extract_mask::Pipeline
    pipeline_diffusivity::Pipeline
    pipeline_slice::Pipeline
    
    # Pipeline layouts
    layout_seed_init::PipelineLayout
    layout_diffuse_step::PipelineLayout
    layout_extract_mask::PipelineLayout
    layout_diffusivity::PipelineLayout
    layout_slice::PipelineLayout
    
    # Descriptor set layouts
    dsl_seed_init::DescriptorSetLayout
    dsl_diffuse_step::DescriptorSetLayout
    dsl_extract_mask::DescriptorSetLayout
    dsl_diffusivity::DescriptorSetLayout
    dsl_slice::DescriptorSetLayout
    
    # Descriptor pool and sets
    descriptor_pool::DescriptorPool
    ds_seed_ping::DescriptorSet          # seed_init writing to ping
    ds_seed_pong::DescriptorSet          # seed_init writing to pong
    ds_diffuse_ping_to_pong::DescriptorSet  # diffuse: read ping, write pong
    ds_diffuse_pong_to_ping::DescriptorSet  # diffuse: read pong, write ping
    ds_extract_from_ping::DescriptorSet     # extract mask from ping
    ds_extract_from_pong::DescriptorSet     # extract mask from pong
    ds_compute_diffusivity::DescriptorSet   # edge→diffusivity conversion
    ds_slice_ping::DescriptorSet            # slice from ping
    ds_slice_pong::DescriptorSet            # slice from pong
    
    # Compute command pool and buffer
    compute_cmd_pool::CommandPool
    compute_cmd_buffer::CommandBuffer
    compute_fence::Fence
    
    # State tracking
    is_initialized::Bool
    diffusivity_uploaded::Bool
    current_buffer_is_ping::Bool   # true = latest data in ping buffer, false = pong
    accumulated_steps::Int         # total diffusion steps since last seed
    
    # Persistent staging buffer for zero-allocation readbacks
    staging_box_buf::Union{Nothing, Buffer}
    staging_box_mem::Union{Nothing, DeviceMemory}
    staging_box_ptr::Ptr{Cvoid}
    staging_box_capacity::Int
    
    # Lesion tracking
    seed_x::Int
    seed_y::Int
    seed_z::Int
    seed_radius::Int
    current_box_min::NTuple{3, Int}
    current_box_max::NTuple{3, Int}
    spacing::NTuple{3, Float64}
    current_s_boost::Float32
    current_edge_params::NTuple{12, Float32}
    
    # Dummy sampler for DescriptorImageInfo
    dummy_sampler::Sampler
    
    # Default constructor with uninitialized fields
    function HeatDiffusionState()
        state = new()
        state.is_initialized = false
        state.diffusivity_uploaded = false
        state.current_buffer_is_ping = true
        state.accumulated_steps = 0
        state.width = 0
        state.height = 0
        state.depth = 0
        
        state.staging_box_buf = nothing
        state.staging_box_mem = nothing
        state.staging_box_ptr = C_NULL
        state.staging_box_capacity = 0
        
        state.seed_x = 0
        state.seed_y = 0
        state.seed_z = 0
        state.seed_radius = 0
        state.current_box_min = (1, 1, 1)
        state.current_box_max = (1, 1, 1)
        state.spacing = (1.0, 1.0, 1.0)
        state.current_s_boost = 1.0f0
        state.current_edge_params = (0.20f0, 0.20f0, 0.20f0, 0.20f0, 0.20f0, 0.20f0, 0.05f0, 0.85f0, 12.0f0, 1.0f0, 0.0f0, 0.0f0)
        return state
    end
end

"""
    _normalize_edge_params(edge_params) -> NTuple{12, Float32}

Normalizes isotropic (4-element) or directional (10- to 12-element) edge parameters
into a unified 12-element Float32 tuple matching the GLSL push constants layout:
(peak_xp, peak_xm, peak_yp, peak_ym, peak_zp, peak_zm, edge_bg, edge_clip, gamma_eff, is_directional, pad1, pad2).
"""
function _normalize_edge_params(edge_params)::NTuple{12, Float32}
    if length(edge_params) == 4
        # Backward compatibility with isotropic 4-tuple: (E_bg, E_scale, gamma_eff, edge_clip)
        E_bg = Float32(edge_params[1])
        E_scale = Float32(edge_params[2])
        gamma_eff = Float32(edge_params[3])
        edge_clip = Float32(edge_params[4])
        E_peak = (E_scale > 0.0f0) ? (E_bg + 1.0f0 / E_scale) : 0.20f0
        return (E_peak, E_peak, E_peak, E_peak, E_peak, E_peak, E_bg, edge_clip, gamma_eff, 0.0f0, 0.35f0, 0.0f0)
    elseif length(edge_params) >= 10
        p_xp = Float32(edge_params[1])
        p_xm = Float32(edge_params[2])
        p_yp = Float32(edge_params[3])
        p_ym = Float32(edge_params[4])
        p_zp = Float32(edge_params[5])
        p_zm = Float32(edge_params[6])
        E_bg = Float32(edge_params[7])
        edge_clip = Float32(edge_params[8])
        gamma_eff = Float32(edge_params[9])
        is_dir = Float32(edge_params[10])
        edge_barrier = length(edge_params) >= 11 && edge_params[11] > 0.05f0 ? Float32(edge_params[11]) : 0.35f0
        is_tubular = length(edge_params) >= 12 ? Float32(edge_params[12]) : 0.0f0
        return (p_xp, p_xm, p_yp, p_ym, p_zp, p_zm, E_bg, edge_clip, gamma_eff, is_dir, edge_barrier, is_tubular)
    else
        return (0.20f0, 0.20f0, 0.20f0, 0.20f0, 0.20f0, 0.20f0, 0.05f0, 0.85f0, 12.0f0, 1.0f0, 0.35f0, 0.0f0)
    end
end

"""
    _compute_spacing_weights(spacing) -> (inv_dx2, inv_dy2, inv_dz2)

Computes normalized inverse-squared spacing weights (w_x, w_y, w_z) such that
the smallest voxel dimension has weight 1.0 and coarser dimensions (e.g. slice thickness)
have weights (s_min / s_d)^2 <= 1.0. This guarantees unconditional numerical stability
for the explicit Euler heat diffusion step while properly scaling cross-slice flux.
"""
function _compute_spacing_weights(spacing::NTuple{3, Real})
    dx, dy, dz = Float32(spacing[1]), Float32(spacing[2]), Float32(spacing[3])
    if dx <= 0.0f0 || dy <= 0.0f0 || dz <= 0.0f0 || !isfinite(dx) || !isfinite(dy) || !isfinite(dz)
        return 1.0f0, 1.0f0, 1.0f0
    end
    min_sp = min(dx, dy, dz)
    inv_dx2 = (min_sp / dx)^2
    inv_dy2 = (min_sp / dy)^2
    inv_dz2 = (min_sp / dz)^2
    return inv_dx2, inv_dy2, inv_dz2
end

"""
    compute_frontier_edge_scale(diffusivity::Array{Float32,3}, cx::Int, cy::Int, cz::Int,
                               R_phys::Real, spacing::NTuple{3,Real};
                               margin_phys::Real=6.0,
                               E_ref::Float32=0.20f0,
                               sensitivity::Float32=1.0f0)::Float32

Samples the dilated frontier shell (outer band of thickness `margin_phys` mm ahead of the
current lesion boundary at radius `R_phys` mm) from `diffusivity`.
Computes mean edge E_frontier and returns the adaptive contrast boost exponent:
    s_boost = clamp((E_ref / max(E_frontier, 0.01))^sensitivity, 0.5, 10.0)
"""
function compute_frontier_edge_scale(diffusivity::Array{Float32,3}, cx::Int, cy::Int, cz::Int,
                                     R_phys::Real, spacing::NTuple{3,Real};
                                     margin_phys::Real=6.0,
                                     E_ref::Float32=0.20f0,
                                     sensitivity::Float32=1.0f0)::Float32
    w, h, d = size(diffusivity)
    dx, dy, dz = Float64.(spacing)
    
    # Convert physical radii to voxel radii
    R_outer_mm = max(3.0, Float64(R_phys))
    R_inner_mm = max(0.0, R_outer_mm - Float64(margin_phys))
    
    rx_out = max(2, round(Int, R_outer_mm / dx))
    ry_out = max(2, round(Int, R_outer_mm / dy))
    rz_out = max(1, round(Int, R_outer_mm / dz))
    
    rx_in = max(0, round(Int, R_inner_mm / dx))
    ry_in = max(0, round(Int, R_inner_mm / dy))
    rz_in = max(0, round(Int, R_inner_mm / dz))
    
    x1, x2 = max(1, cx - rx_out), min(w, cx + rx_out)
    y1, y2 = max(1, cy - ry_out), min(h, cy + ry_out)
    z1, z2 = max(1, cz - rz_out), min(d, cz + rz_out)
    
    inv_rx_out2 = 1.0 / (rx_out^2)
    inv_ry_out2 = 1.0 / (ry_out^2)
    inv_rz_out2 = 1.0 / (rz_out^2)
    
    inv_rx_in2 = rx_in > 0 ? 1.0 / (rx_in^2) : 0.0
    inv_ry_in2 = ry_in > 0 ? 1.0 / (ry_in^2) : 0.0
    inv_rz_in2 = rz_in > 0 ? 1.0 / (rz_in^2) : 0.0
    
    sum_E = 0.0f0
    count_vox = 0
    
    step_x = (x2 - x1 > 40) ? 2 : 1
    step_y = (y2 - y1 > 40) ? 2 : 1
    step_z = (z2 - z1 > 40) ? 2 : 1
    
    for z in z1:step_z:z2
        dz_out2 = (z - cz)^2 * inv_rz_out2
        dz_out2 > 1.0 && continue
        dz_in2 = (z - cz)^2 * inv_rz_in2
        
        for y in y1:step_y:y2
            dy_out2 = (y - cy)^2 * inv_ry_out2
            (dz_out2 + dy_out2) > 1.0 && continue
            dy_in2 = (y - cy)^2 * inv_ry_in2
            
            for x in x1:step_x:x2
                dx_out2 = (x - cx)^2 * inv_rx_out2
                d_out2 = dz_out2 + dy_out2 + dx_out2
                d_out2 > 1.0 && continue
                
                # Check if outside inner core (frontier shell)
                if rx_in > 0 && ry_in > 0 && rz_in > 0
                    d_in2 = dz_in2 + dy_in2 + (x - cx)^2 * inv_rx_in2
                    d_in2 < 1.0 && continue
                end
                
                D_val = diffusivity[x, y, z]
                # Invert D to E: E = -ln(max(D - 1e-5, 1e-7)) / 10.096
                E_val = -log(clamp(D_val, 1.0f-7, 1.0f0)) / 10.095886f0
                sum_E += E_val
                count_vox += 1
            end
        end
    end
    
    count_vox == 0 && return 1.0f0
    mean_E = sum_E / count_vox
    mean_E_clamped = max(mean_E, 0.01f0)
    
    s_raw = (E_ref / mean_E_clamped)^sensitivity
    s_boost = clamp(s_raw, 0.5f0, 10.0f0)
    return Float32(s_boost)
end

"""
    compute_local_relative_edge_params(diffusivity::Array{Float32,3}, cx::Int, cy::Int, cz::Int,
                                       R_phys::Real, spacing::NTuple{3,Real};
                                       sensitivity::Float32=1.0f0,
                                       edge_clip::Float32=0.85f0)::NTuple{4, Float32}

Analyzes the local neighborhood around the seed (cx, cy, cz) to compute relative edge normalization parameters:
1. Samples the inner core (r <= 4.0 mm) to find the baseline interior edge strength E_bg
   (using 25th percentile of core samples, ensuring baseline represents pure lesion interior).
2. Samples the surrounding vicinity shell (4.0 mm < r <= R_phys mm) to determine the peak local edge E_peak
   (using 95th percentile of shell samples to resist noise spikes).
3. Computes:
   - E_bg: baseline interior edge
   - E_scale: 1.0 / max(E_peak - E_bg, 0.02)
   - gamma_eff: 12.0 * sensitivity (clamped to [8.0, 24.0])
   - edge_clip: relative edge threshold (default 0.85) above which D_eff is forced to 1.0e-5.

Inside the lesion core where E <= E_bg, E_rel = 0.0 -> D_eff = 1.0 (free internal flow independent
of absolute background soft tissue attenuation).
At the strongest edge in the immediate vicinity (E -> E_peak), E_rel -> 1.0 -> D_eff = 1.0e-5
(acts as an impenetrable barrier, even for weak soft-tissue boundaries).
"""
function compute_local_relative_edge_params(diffusivity::Array{Float32,3}, cx::Int, cy::Int, cz::Int,
                                            R_phys::Real, spacing::NTuple{3,Real};
                                            sensitivity::Float32=1.0f0,
                                            edge_clip::Float32=0.85f0)::NTuple{4, Float32}
    w, h, d = size(diffusivity)
    dx, dy, dz = Float64.(spacing)
    
    R_core_mm = 4.0
    R_outer_mm = max(8.0, Float64(R_phys))
    
    rx_out = max(2, round(Int, R_outer_mm / dx))
    ry_out = max(2, round(Int, R_outer_mm / dy))
    rz_out = max(1, round(Int, R_outer_mm / dz))
    
    rx_core = max(1, round(Int, R_core_mm / dx))
    ry_core = max(1, round(Int, R_core_mm / dy))
    rz_core = max(1, round(Int, R_core_mm / dz))
    
    x1, x2 = max(1, cx - rx_out), min(w, cx + rx_out)
    y1, y2 = max(1, cy - ry_out), min(h, cy + ry_out)
    z1, z2 = max(1, cz - rz_out), min(d, cz + rz_out)
    
    inv_rx_out2 = 1.0 / (rx_out^2)
    inv_ry_out2 = 1.0 / (ry_out^2)
    inv_rz_out2 = 1.0 / (rz_out^2)
    
    inv_rx_core2 = 1.0 / (rx_core^2)
    inv_ry_core2 = 1.0 / (ry_core^2)
    inv_rz_core2 = 1.0 / (rz_core^2)
    
    core_samples = Float32[]
    shell_samples = Float32[]
    sizehint!(core_samples, 200)
    sizehint!(shell_samples, 1000)
    
    step_x = (x2 - x1 > 40) ? 2 : 1
    step_y = (y2 - y1 > 40) ? 2 : 1
    step_z = (z2 - z1 > 40) ? 2 : 1
    
    for z in z1:step_z:z2
        dz_out2 = (z - cz)^2 * inv_rz_out2
        dz_out2 > 1.0 && continue
        dz_core2 = (z - cz)^2 * inv_rz_core2
        
        for y in y1:step_y:y2
            dy_out2 = (y - cy)^2 * inv_ry_out2
            (dz_out2 + dy_out2) > 1.0 && continue
            dy_core2 = (y - cy)^2 * inv_ry_core2
            
            for x in x1:step_x:x2
                dx_out2 = (x - cx)^2 * inv_rx_out2
                d_out2 = dz_out2 + dy_out2 + dx_out2
                d_out2 > 1.0 && continue
                
                D_val = diffusivity[x, y, z]
                # E = -ln(D) / 10.095886
                E_val = -log(clamp(D_val, 1.0f-5, 1.0f0)) * 0.09905025f0
                
                # Check if in core
                d_core2 = dz_core2 + dy_core2 + (x - cx)^2 * inv_rx_core2
                if d_core2 <= 1.0
                    push!(core_samples, E_val)
                else
                    push!(shell_samples, E_val)
                end
            end
        end
    end
    
    E_bg = if !isempty(core_samples)
        sort!(core_samples)
        idx = max(1, round(Int, 0.25 * length(core_samples)))
        core_samples[idx]
    else
        0.05f0
    end
    
    E_peak = if !isempty(shell_samples)
        sort!(shell_samples)
        idx = min(length(shell_samples), max(1, round(Int, 0.95 * length(shell_samples))))
        shell_samples[idx]
    else
        0.20f0
    end
    
    delta_E = max(E_peak - E_bg, 0.02f0)
    E_scale = 1.0f0 / delta_E
    gamma_eff = clamp(12.0f0 * sensitivity, 8.0f0, 24.0f0)
    
    return (Float32(E_bg), Float32(E_scale), Float32(gamma_eff), Float32(edge_clip))
end

"""
    compute_directional_relative_edge_params(diffusivity::Array{Float32,3}, cx::Int, cy::Int, cz::Int,
                                             R_phys::Real, spacing::NTuple{3,Real};
                                             sensitivity::Float32=1.0f0,
                                             edge_clip::Float32=0.85f0)::NTuple{12, Float32}

Analyzes the local 3D neighborhood around the seed (cx, cy, cz) to compute directional
relative edge normalization parameters across 6 cardinal sectors (+X, -X, +Y, -Y, +Z, -Z):
1. Samples E_seed = E(cx, cy, cz) and traces cardinal rays to detect circumferential walls.
2. If in a narrow tubular structure (r_wall <= 3.5 mm, e.g. ureter, vessel):
   - Automatically adapts R_core to fit strictly inside the lumen.
   - Anchors baseline E_bg to the lumen center.
   - Sets dynamic barrier ceiling E_barrier = max(0.40, E_seed + 0.15), preventing the seed
     from freezing out while preserving impenetrable circumferential side walls.
   - Restricts longitudinal sampling to a narrow corridor along the tube axis, preventing
     wide-angle cone contamination from surrounding exterior tissue.
3. If in standard rounded tissue/tumor:
   - Samples 6 cardinal cones to resolve asymmetric interfaces (e.g. diaphragm).
4. Returns (peak_+X, peak_-X, peak_+Y, peak_-Y, peak_+Z, peak_-Z, E_bg, edge_clip, gamma_eff, is_directional, E_barrier, is_tubular).
"""
function compute_directional_relative_edge_params(diffusivity::Array{Float32,3}, cx::Int, cy::Int, cz::Int,
                                                  R_phys::Real, spacing::NTuple{3,Real};
                                                  sensitivity::Float32=1.0f0,
                                                  edge_clip::Float32=0.85f0)::NTuple{12, Float32}
    w, h, d = size(diffusivity)
    dx, dy, dz = Float64.(spacing)
    
    # 1. Edge at seed location
    D_seed = diffusivity[cx, cy, cz]
    E_seed = -log(clamp(D_seed, 1.0f-5, 1.0f0)) * 0.09905025f0
    
    # 2. Analyze candidate axis A in (3=Z, 1=X, 2=Y) for narrow tubular channel
    # (Breakthrough 7: Lumen Ridge/Valley Follower with Curvature-Adaptive Corridor Tracking)
    best_axis = 3
    best_reach = 0.0
    best_lumen_samples = Float32[]
    best_wall_dist = 999.0
    best_wall_edges = Float32[]
    
    for axis in (3, 1, 2)
        step_vox = (axis == 3) ? dz : ((axis == 2) ? dy : dx)
        max_steps = max(3, round(Int, 35.0 / step_vox))
        
        s_r1 = (axis == 1) ? max(1, round(Int, 2.5 / dy)) : max(1, round(Int, 2.5 / dx))
        s_r2 = (axis == 3) ? max(1, round(Int, 2.5 / dy)) : max(1, round(Int, 2.5 / dz))
        
        # Initial 2D centering at seed slice
        min_e_seed = E_seed
        best_pt_seed = (cx, cy, cz)
        
        if axis == 3
            for y in max(1, cy - s_r2):min(h, cy + s_r2)
                for x in max(1, cx - s_r1):min(w, cx + s_r1)
                    e = -log(clamp(diffusivity[x, y, cz], 1.0f-5, 1.0f0)) * 0.09905025f0
                    if e < min_e_seed; min_e_seed = e; best_pt_seed = (x, y, cz); end
                end
            end
        elseif axis == 1
            for z in max(1, cz - s_r2):min(d, cz + s_r2)
                for y in max(1, cy - s_r1):min(h, cy + s_r1)
                    e = -log(clamp(diffusivity[cx, y, z], 1.0f-5, 1.0f0)) * 0.09905025f0
                    if e < min_e_seed; min_e_seed = e; best_pt_seed = (cx, y, z); end
                end
            end
        else # axis == 2
            for z in max(1, cz - s_r2):min(d, cz + s_r2)
                for x in max(1, cx - s_r1):min(w, cx + s_r1)
                    e = -log(clamp(diffusivity[x, cy, z], 1.0f-5, 1.0f0)) * 0.09905025f0
                    if e < min_e_seed; min_e_seed = e; best_pt_seed = (x, cy, z); end
                end
            end
        end
        
        # Measure transverse wall distance and peak wall edges at centered seed
        t_wall_dist = fill(8.0, 4)
        t_wall_edges = Float32[]
        wall_thresh = max(0.35f0, min_e_seed + 0.12f0)
        
        if axis == 3
            dirs = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0))
            for k in 1:4
                dir = dirs[k]
                max_ray_e = 0.0f0
                for s in 1:16
                    d_mm = s * 0.5
                    px = clamp(round(Int, best_pt_seed[1] + dir[1] * d_mm / dx), 1, w)
                    py = clamp(round(Int, best_pt_seed[2] + dir[2] * d_mm / dy), 1, h)
                    e = -log(clamp(diffusivity[px, py, cz], 1.0f-5, 1.0f0)) * 0.09905025f0
                    max_ray_e = max(max_ray_e, e)
                    if (e >= wall_thresh || e >= 0.85f0) && t_wall_dist[k] == 8.0
                        t_wall_dist[k] = d_mm
                    end
                end
                push!(t_wall_edges, max_ray_e)
            end
        elseif axis == 1
            dirs = ((0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
            for k in 1:4
                dir = dirs[k]
                max_ray_e = 0.0f0
                for s in 1:16
                    d_mm = s * 0.5
                    py = clamp(round(Int, best_pt_seed[2] + dir[2] * d_mm / dy), 1, h)
                    pz = clamp(round(Int, best_pt_seed[3] + dir[3] * d_mm / dz), 1, d)
                    e = -log(clamp(diffusivity[best_pt_seed[1], py, pz], 1.0f-5, 1.0f0)) * 0.09905025f0
                    max_ray_e = max(max_ray_e, e)
                    if (e >= wall_thresh || e >= 0.85f0) && t_wall_dist[k] == 8.0
                        t_wall_dist[k] = d_mm
                    end
                end
                push!(t_wall_edges, max_ray_e)
            end
        else # axis == 2
            dirs = ((1, 0, 0), (-1, 0, 0), (0, 0, 1), (0, 0, -1))
            for k in 1:4
                dir = dirs[k]
                max_ray_e = 0.0f0
                for s in 1:16
                    d_mm = s * 0.5
                    px = clamp(round(Int, best_pt_seed[1] + dir[1] * d_mm / dx), 1, w)
                    pz = clamp(round(Int, best_pt_seed[3] + dir[3] * d_mm / dz), 1, d)
                    e = -log(clamp(diffusivity[px, best_pt_seed[2], pz], 1.0f-5, 1.0f0)) * 0.09905025f0
                    max_ray_e = max(max_ray_e, e)
                    if (e >= wall_thresh || e >= 0.85f0) && t_wall_dist[k] == 8.0
                        t_wall_dist[k] = d_mm
                    end
                end
                push!(t_wall_edges, max_ray_e)
            end
        end
        
        min_wall_mm = minimum(t_wall_dist)
        
        if min_wall_mm <= 4.0
            lumen_samples = Float32[min_e_seed]
            wall_samples = copy(t_wall_edges)
            reach_fwd = 0.0
            reach_bwd = 0.0
            
            # Forward trace
            cur_p = best_pt_seed
            for s in 1:max_steps
                if axis == 3
                    z = cur_p[3] + 1; z > d && break
                    bx, by, be = cur_p[1], cur_p[2], 1.0f0
                    for y in max(1, cur_p[2] - s_r2):min(h, cur_p[2] + s_r2)
                        for x in max(1, cur_p[1] - s_r1):min(w, cur_p[1] + s_r1)
                            e = -log(clamp(diffusivity[x, y, z], 1.0f-5, 1.0f0)) * 0.09905025f0
                            if e < be; be = e; bx, by = x, y; end
                        end
                    end
                    if be >= 0.88f0 || be >= min_e_seed + 0.25f0 || (min_e_seed < 0.30f0 && be >= 0.40f0); break; end
                    cur_p = (bx, by, z); push!(lumen_samples, be); reach_fwd += dz
                elseif axis == 1
                    x = cur_p[1] + 1; x > w && break
                    by, bz, be = cur_p[2], cur_p[3], 1.0f0
                    for z in max(1, cur_p[3] - s_r2):min(d, cur_p[3] + s_r2)
                        for y in max(1, cur_p[2] - s_r1):min(h, cur_p[2] + s_r1)
                            e = -log(clamp(diffusivity[x, y, z], 1.0f-5, 1.0f0)) * 0.09905025f0
                            if e < be; be = e; by, bz = y, z; end
                        end
                    end
                    if be >= 0.88f0 || be >= min_e_seed + 0.25f0 || (min_e_seed < 0.30f0 && be >= 0.40f0); break; end
                    cur_p = (x, by, bz); push!(lumen_samples, be); reach_fwd += dx
                else # axis == 2
                    y = cur_p[2] + 1; y > h && break
                    bx, bz, be = cur_p[1], cur_p[3], 1.0f0
                    for z in max(1, cur_p[3] - s_r2):min(d, cur_p[3] + s_r2)
                        for x in max(1, cur_p[1] - s_r1):min(w, cur_p[1] + s_r1)
                            e = -log(clamp(diffusivity[x, y, z], 1.0f-5, 1.0f0)) * 0.09905025f0
                            if e < be; be = e; bx, bz = x, z; end
                        end
                    end
                    if be >= 0.88f0 || be >= min_e_seed + 0.25f0 || (min_e_seed < 0.30f0 && be >= 0.40f0); break; end
                    cur_p = (bx, y, bz); push!(lumen_samples, be); reach_fwd += dy
                end
            end
            
            # Backward trace
            cur_p = best_pt_seed
            for s in 1:max_steps
                if axis == 3
                    z = cur_p[3] - 1; z < 1 && break
                    bx, by, be = cur_p[1], cur_p[2], 1.0f0
                    for y in max(1, cur_p[2] - s_r2):min(h, cur_p[2] + s_r2)
                        for x in max(1, cur_p[1] - s_r1):min(w, cur_p[1] + s_r1)
                            e = -log(clamp(diffusivity[x, y, z], 1.0f-5, 1.0f0)) * 0.09905025f0
                            if e < be; be = e; bx, by = x, y; end
                        end
                    end
                    if be >= 0.88f0 || be >= min_e_seed + 0.25f0 || (min_e_seed < 0.30f0 && be >= 0.40f0); break; end
                    cur_p = (bx, by, z); push!(lumen_samples, be); reach_bwd += dz
                elseif axis == 1
                    x = cur_p[1] - 1; x < 1 && break
                    by, bz, be = cur_p[2], cur_p[3], 1.0f0
                    for z in max(1, cur_p[3] - s_r2):min(d, cur_p[3] + s_r2)
                        for y in max(1, cur_p[2] - s_r1):min(h, cur_p[2] + s_r1)
                            e = -log(clamp(diffusivity[x, y, z], 1.0f-5, 1.0f0)) * 0.09905025f0
                            if e < be; be = e; by, bz = y, z; end
                        end
                    end
                    if be >= 0.88f0 || be >= min_e_seed + 0.25f0 || (min_e_seed < 0.30f0 && be >= 0.40f0); break; end
                    cur_p = (x, by, bz); push!(lumen_samples, be); reach_bwd += dx
                else # axis == 2
                    y = cur_p[2] - 1; y < 1 && break
                    bx, bz, be = cur_p[1], cur_p[3], 1.0f0
                    for z in max(1, cur_p[3] - s_r2):min(d, cur_p[3] + s_r2)
                        for x in max(1, cur_p[1] - s_r1):min(w, cur_p[1] + s_r1)
                            e = -log(clamp(diffusivity[x, y, z], 1.0f-5, 1.0f0)) * 0.09905025f0
                            if e < be; be = e; bx, bz = x, z; end
                        end
                    end
                    if be >= 0.88f0 || be >= min_e_seed + 0.25f0 || (min_e_seed < 0.30f0 && be >= 0.40f0); break; end
                    cur_p = (bx, y, bz); push!(lumen_samples, be); reach_bwd += dy
                end
            end
            
            tot_reach = reach_fwd + reach_bwd
            if tot_reach > best_reach
                best_reach = tot_reach
                best_axis = axis
                best_lumen_samples = lumen_samples
                best_wall_dist = min_wall_mm
                best_wall_edges = wall_samples
            end
        end
    end
    
    is_tubular = (best_wall_dist <= 4.0) && (best_reach >= 8.0)
    
    if is_tubular
        sort!(best_lumen_samples)
        p50_lumen = best_lumen_samples[min(length(best_lumen_samples), max(1, round(Int, 0.50 * length(best_lumen_samples))))]
        p85_lumen = best_lumen_samples[min(length(best_lumen_samples), max(1, round(Int, 0.85 * length(best_lumen_samples))))]
        E_bg = if E_seed < 0.30f0
            p50_lumen
        else
            min(0.80f0, max(E_seed, p85_lumen))
        end
        
        E_barrier = if !isempty(best_wall_edges)
            sort!(best_wall_edges)
            p75_wall = best_wall_edges[min(length(best_wall_edges), max(1, round(Int, 0.75 * length(best_wall_edges))))]
            max(0.38f0, min(0.96f0, p75_wall - 0.02f0))
        else
            max(0.40f0, E_seed + 0.15f0)
        end
        E_barrier = max(E_barrier, E_bg + 0.08f0)
        
        p_long = min(E_bg + 0.04f0, E_barrier - 0.02f0)
        p_trans = max(0.40f0, E_barrier)
        
        p_xp, p_xm, p_yp, p_ym, p_zp, p_zm = if best_axis == 3
            p_trans, p_trans, p_trans, p_trans, p_long, p_long
        elseif best_axis == 1
            p_long, p_long, p_trans, p_trans, p_trans, p_trans
        else
            p_trans, p_trans, p_long, p_long, p_trans, p_trans
        end
        
        gamma_eff = clamp(10.0f0 * sensitivity, 8.0f0, 20.0f0)
        return (Float32(p_xp), Float32(p_xm), Float32(p_yp), Float32(p_ym), Float32(p_zp), Float32(p_zm),
                Float32(E_bg), Float32(edge_clip), Float32(gamma_eff), 1.0f0, Float32(E_barrier), 1.0f0)
    else
        # Standard directional relative edge normalization (Breakthrough 5)
        R_outer_mm = max(8.0, Float64(R_phys))
        R_core_mm = 4.0
        
        rx_out = max(2, round(Int, R_outer_mm / dx))
        ry_out = max(2, round(Int, R_outer_mm / dy))
        rz_out = max(1, round(Int, R_outer_mm / dz))
        
        rx_core = max(1, round(Int, R_core_mm / dx))
        ry_core = max(1, round(Int, R_core_mm / dy))
        rz_core = max(1, round(Int, R_core_mm / dz))
        
        x1, x2 = max(1, cx - rx_out), min(w, cx + rx_out)
        y1, y2 = max(1, cy - ry_out), min(h, cy + ry_out)
        z1, z2 = max(1, cz - rz_out), min(d, cz + rz_out)
        
        inv_rx_out2 = 1.0 / (rx_out^2)
        inv_ry_out2 = 1.0 / (ry_out^2)
        inv_rz_out2 = 1.0 / (rz_out^2)
        
        inv_rx_core2 = 1.0 / (rx_core^2)
        inv_ry_core2 = 1.0 / (ry_core^2)
        inv_rz_core2 = 1.0 / (rz_core^2)
        
        core_samples = Float32[]
        shell_samples = Float32[]
        shell_xp = Float32[]; shell_xm = Float32[]
        shell_yp = Float32[]; shell_ym = Float32[]
        shell_zp = Float32[]; shell_zm = Float32[]
        
        sizehint!(core_samples, 200)
        sizehint!(shell_samples, 1000)
        sizehint!(shell_xp, 200); sizehint!(shell_xm, 200)
        sizehint!(shell_yp, 200); sizehint!(shell_ym, 200)
        sizehint!(shell_zp, 200); sizehint!(shell_zm, 200)
        
        step_x = (x2 - x1 > 40) ? 2 : 1
        step_y = (y2 - y1 > 40) ? 2 : 1
        step_z = (z2 - z1 > 40) ? 2 : 1
        
        for z in z1:step_z:z2
            dz_out2 = (z - cz)^2 * inv_rz_out2
            dz_out2 > 1.0 && continue
            dz_core2 = (z - cz)^2 * inv_rz_core2
            dz_mm = (Float64(z) - Float64(cz)) * dz
            abs_dz = abs(dz_mm)
            
            for y in y1:step_y:y2
                dy_out2 = (y - cy)^2 * inv_ry_out2
                (dz_out2 + dy_out2) > 1.0 && continue
                dy_core2 = (y - cy)^2 * inv_ry_core2
                dy_mm = (Float64(y) - Float64(cy)) * dy
                abs_dy = abs(dy_mm)
                
                for x in x1:step_x:x2
                    dx_out2 = (x - cx)^2 * inv_rx_out2
                    (dz_out2 + dy_out2 + dx_out2) > 1.0 && continue
                    
                    D_val = diffusivity[x, y, z]
                    E_val = -log(clamp(D_val, 1.0f-5, 1.0f0)) * 0.09905025f0
                    
                    d_core2 = dz_core2 + dy_core2 + (x - cx)^2 * inv_rx_core2
                    if d_core2 <= 1.0
                        push!(core_samples, E_val)
                    else
                        push!(shell_samples, E_val)
                        dx_mm = (Float64(x) - Float64(cx)) * dx
                        abs_dx = abs(dx_mm)
                        
                        if abs_dx >= abs_dy && abs_dx >= abs_dz
                            (dx_mm >= 0.0) ? push!(shell_xp, E_val) : push!(shell_xm, E_val)
                        elseif abs_dy >= abs_dx && abs_dy >= abs_dz
                            (dy_mm >= 0.0) ? push!(shell_yp, E_val) : push!(shell_ym, E_val)
                        else
                            (dz_mm >= 0.0) ? push!(shell_zp, E_val) : push!(shell_zm, E_val)
                        end
                    end
                end
            end
        end
        
        E_bg = if !isempty(core_samples)
            sort!(core_samples)
            idx = max(1, round(Int, 0.25 * length(core_samples)))
            min(E_seed, core_samples[idx])
        else
            min(E_seed, 0.05f0)
        end
        
        global_shell_peak = if !isempty(shell_samples)
            sort!(shell_samples)
            idx = min(length(shell_samples), max(1, round(Int, 0.95 * length(shell_samples))))
            shell_samples[idx]
        else
            0.20f0
        end
        
        function _calc_peak(samples::Vector{Float32})
            if length(samples) >= 5
                sort!(samples)
                idx = min(length(samples), max(1, round(Int, 0.95 * length(samples))))
                return max(samples[idx], E_bg + 0.02f0)
            else
                return max(global_shell_peak, E_bg + 0.02f0)
            end
        end
        
        p_xp = _calc_peak(shell_xp)
        p_xm = _calc_peak(shell_xm)
        p_yp = _calc_peak(shell_yp)
        p_ym = _calc_peak(shell_ym)
        p_zp = _calc_peak(shell_zp)
        p_zm = _calc_peak(shell_zm)
        
        E_barrier = max(0.35f0, E_seed + 0.10f0)
        gamma_eff = clamp(12.0f0 * sensitivity, 8.0f0, 24.0f0)
        return (Float32(p_xp), Float32(p_xm), Float32(p_yp), Float32(p_ym), Float32(p_zp), Float32(p_zm),
                Float32(E_bg), Float32(edge_clip), Float32(gamma_eff), 1.0f0, Float32(E_barrier), 0.0f0)
    end
end

# ─── Shader loading ──────────────────────────────────────────────────────

"""
    get_shader_dir() -> String

Dynamically resolves the directory containing compiled Vulkan SPIR-V (.spv) shaders.
Searches in order:
1. `Sys.BINDIR/shaders` (installed app binary directory)
2. `app_root/shaders` (installed app root directory)
3. `app_root/src/display/Vulkan/shaders` (relative to app bundle)
4. `%APPDATA%/MedEye3D/shaders`
5. Local user app data paths
6. Source directory relative to `@__DIR__`
"""
function get_shader_dir()::String
    app_root = dirname(Sys.BINDIR)
    candidates = [
        joinpath(Sys.BINDIR, "shaders"),
        joinpath(app_root, "shaders"),
        joinpath(app_root, "src", "display", "Vulkan", "shaders"),
        joinpath(get(ENV, "APPDATA", ""), "MedEye3D", "shaders"),
        joinpath(homedir(), "AppData", "Local", "Programs", "MedEye3D", "shaders"),
        joinpath(homedir(), "AppData", "Local", "Programs", "MedEye3D", "bin", "shaders"),
        normpath(joinpath(@__DIR__, "shaders")),
        normpath(joinpath(@__DIR__, "..", "..", "display", "Vulkan", "shaders")),
        normpath(joinpath(@__DIR__, "..", "src", "display", "Vulkan", "shaders"))
    ]
    for c in candidates
        try
            if isdir(c) && isfile(joinpath(c, "seed_init.spv"))
                return c
            end
        catch
        end
    end
    for c in candidates
        try
            if isdir(c)
                return c
            end
        catch
        end
    end
    return normpath(joinpath(@__DIR__, "shaders"))
end

"""
    find_shader_file(filename::String) -> String

Finds the absolute path to a specific shader file (.spv or .comp).
"""
function find_shader_file(filename::String)::String
    sdir = get_shader_dir()
    p = joinpath(sdir, filename)
    if isfile(p)
        return p
    end
    app_root = dirname(Sys.BINDIR)
    candidates = [
        joinpath(Sys.BINDIR, "shaders", filename),
        joinpath(app_root, "shaders", filename),
        joinpath(app_root, "src", "display", "Vulkan", "shaders", filename),
        joinpath(get(ENV, "APPDATA", ""), "MedEye3D", "shaders", filename),
        joinpath(homedir(), "AppData", "Local", "Programs", "MedEye3D", "shaders", filename),
        joinpath(homedir(), "AppData", "Local", "Programs", "MedEye3D", "bin", "shaders", filename),
        normpath(joinpath(@__DIR__, "shaders", filename)),
        normpath(joinpath(@__DIR__, "..", "..", "display", "Vulkan", "shaders", filename))
    ]
    for c in candidates
        try
            if isfile(c)
                return c
            end
        catch
        end
    end
    return p
end

const SHADER_DIR = normpath(joinpath(@__DIR__, "shaders"))

function load_shader_module(device::Device, filename::String)
    spv_path = find_shader_file(filename)
    local code::Vector{UInt8}
    if isfile(spv_path)
        code = read(spv_path)
    else
        # Fallback: check if .comp file exists and compile dynamically
        comp_name = replace(filename, r"\.spv$" => ".comp")
        comp_path = find_shader_file(comp_name)
        if isfile(comp_path)
            @info "[VulkanHeatDiffusion] Compiling shader from source: $comp_path"
            code = compile_glsl_to_spirv(:comp, read(comp_path, String))
        else
            error("SPIR-V shader not found: $spv_path (get_shader_dir() = $(get_shader_dir()))")
        end
    end
    # Vulkan expects UInt32 aligned code
    code_u32 = reinterpret(UInt32, code)
    create_info = ShaderModuleCreateInfo(sizeof(code), code_u32)
    shader_mod = unwrap(create_shader_module(device, create_info))
    return shader_mod
end

# ─── 3D Image creation helpers ───────────────────────────────────────────

function create_storage_image_3d(device::Device, pdev::PhysicalDevice, w::Int, h::Int, d::Int;
                                  usage_flags = IMAGE_USAGE_STORAGE_BIT | IMAGE_USAGE_TRANSFER_SRC_BIT | IMAGE_USAGE_TRANSFER_DST_BIT)
    create_info = ImageCreateInfo(
        IMAGE_TYPE_3D,
        FORMAT_R32_SFLOAT,
        Extent3D(w, h, d),
        1,  # mip levels
        1,  # array layers
        SAMPLE_COUNT_1_BIT,
        IMAGE_TILING_OPTIMAL,
        usage_flags,
        SHARING_MODE_EXCLUSIVE,
        UInt32[],  # queue family indices
        IMAGE_LAYOUT_UNDEFINED
    )
    img = unwrap(create_image(device, create_info))
    
    # Allocate device memory
    mem_reqs = get_image_memory_requirements(device, img)
    mem_props = get_physical_device_memory_properties(pdev)
    
    mem_type_idx = UInt32(0)
    found = false
    local mem
    for i in 0:(mem_props.memory_type_count - 1)
        if (mem_reqs.memory_type_bits & (1 << i)) != 0 &&
           (mem_props.memory_types[i + 1].property_flags & MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0
            try
                alloc_info = MemoryAllocateInfo(mem_reqs.size, UInt32(i))
                mem = unwrap(allocate_memory(device, alloc_info))
                mem_type_idx = UInt32(i)
                found = true
                break
            catch e
                if e isa Vulkan.VulkanError && e.code == Vulkan.ERROR_OUT_OF_DEVICE_MEMORY
                    @warn "Memory allocation failed on heap $(i), trying next suitable heap..."
                    continue
                end
                rethrow(e)
            end
        end
    end
    if !found
        error("Failed to find suitable memory type or out of memory for 3D storage image")
    end
    unwrap(bind_image_memory(device, img, mem, 0))
    
    # Create image view
    view_info = ImageViewCreateInfo(
        img,
        IMAGE_VIEW_TYPE_3D,
        FORMAT_R32_SFLOAT,
        ComponentMapping(
            COMPONENT_SWIZZLE_IDENTITY,
            COMPONENT_SWIZZLE_IDENTITY,
            COMPONENT_SWIZZLE_IDENTITY,
            COMPONENT_SWIZZLE_IDENTITY
        ),
        ImageSubresourceRange(IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1)
    )
    view = unwrap(create_image_view(device, view_info))
    
    return img, mem, view
end

# ─── Initialization ─────────────────────────────────────────────────────

"""
    init_heat_diffusion!(state, ctx, w, h, d)

Initialize all Vulkan resources for heat diffusion on a volume of size (w, h, d).
`ctx` must have fields: `device`, `physical_device`, `graphics_queue`, `queue_family_index`.
"""
function init_heat_diffusion!(state::HeatDiffusionState, ctx, w::Int, h::Int, d::Int)
    if state.is_initialized
        destroy_heat_diffusion!(state, ctx)
    end
    
    state.width = w
    state.height = h
    state.depth = d
    
    dev = ctx.device
    pdev = ctx.physical_device
    
    @info "[HeatDiffusion] Initializing for volume $(w)×$(h)×$(d)..."
    
    # 1. Create 3D storage images
    state.img_diffusivity, state.mem_diffusivity, state.view_diffusivity = create_storage_image_3d(dev, pdev, w, h, d)
    state.img_u_ping, state.mem_u_ping, state.view_u_ping = create_storage_image_3d(dev, pdev, w, h, d)
    state.img_u_pong, state.mem_u_pong, state.view_u_pong = create_storage_image_3d(dev, pdev, w, h, d)
    state.img_mask, state.mem_mask, state.view_mask = create_storage_image_3d(dev, pdev, w, h, d)
    
    # 2. Load shader modules
    mod_seed = load_shader_module(dev, "seed_init.spv")
    mod_diffuse = load_shader_module(dev, "diffuse_step.spv")
    mod_extract = load_shader_module(dev, "extract_mask.spv")
    mod_diff = load_shader_module(dev, "diffusivity.spv")
    mod_slice = load_shader_module(dev, "slice_heat_to_texture.spv")
    
    # 3. Create descriptor set layouts
    # seed_init: 1 storage image (writeonly)
    bindings_seed = [DescriptorSetLayoutBinding(0, DESCRIPTOR_TYPE_STORAGE_IMAGE, SHADER_STAGE_COMPUTE_BIT; descriptor_count=1)]
    state.dsl_seed_init = unwrap(create_descriptor_set_layout(dev, DescriptorSetLayoutCreateInfo(bindings_seed)))
    
    # diffuse_step: 3 storage images (D readonly, u_in readonly, u_out writeonly)
    bindings_diffuse = [
        DescriptorSetLayoutBinding(0, DESCRIPTOR_TYPE_STORAGE_IMAGE, SHADER_STAGE_COMPUTE_BIT; descriptor_count=1),
        DescriptorSetLayoutBinding(1, DESCRIPTOR_TYPE_STORAGE_IMAGE, SHADER_STAGE_COMPUTE_BIT; descriptor_count=1),
        DescriptorSetLayoutBinding(2, DESCRIPTOR_TYPE_STORAGE_IMAGE, SHADER_STAGE_COMPUTE_BIT; descriptor_count=1)
    ]
    state.dsl_diffuse_step = unwrap(create_descriptor_set_layout(dev, DescriptorSetLayoutCreateInfo(bindings_diffuse)))
    
    # extract_mask: 2 storage images (u readonly, mask writeonly)
    bindings_extract = [
        DescriptorSetLayoutBinding(0, DESCRIPTOR_TYPE_STORAGE_IMAGE, SHADER_STAGE_COMPUTE_BIT; descriptor_count=1),
        DescriptorSetLayoutBinding(1, DESCRIPTOR_TYPE_STORAGE_IMAGE, SHADER_STAGE_COMPUTE_BIT; descriptor_count=1)
    ]
    state.dsl_extract_mask = unwrap(create_descriptor_set_layout(dev, DescriptorSetLayoutCreateInfo(bindings_extract)))
    
    # diffusivity: 2 storage images (edge readonly, diffusivity writeonly)
    state.dsl_diffusivity = unwrap(create_descriptor_set_layout(dev, DescriptorSetLayoutCreateInfo(bindings_extract)))

    # slice: 2 storage images (u readonly, tex writeonly)
    state.dsl_slice = unwrap(create_descriptor_set_layout(dev, DescriptorSetLayoutCreateInfo(bindings_extract)))
    
    # 4. Create pipeline layouts with push constants
    # seed_init: push 2x ivec4 + vec4 (48 bytes)
    pc_seed = [PushConstantRange(SHADER_STAGE_COMPUTE_BIT, 0, 48)]
    state.layout_seed_init = unwrap(create_pipeline_layout(dev, PipelineLayoutCreateInfo([state.dsl_seed_init], pc_seed)))
    
    # diffuse_step: push float dt + box... + inv_d2 + directional edge_params (112 bytes)
    pc_diffuse = [PushConstantRange(SHADER_STAGE_COMPUTE_BIT, 0, sizeof(DiffuseStepPC))]
    state.layout_diffuse_step = unwrap(create_pipeline_layout(dev, PipelineLayoutCreateInfo([state.dsl_diffuse_step], pc_diffuse)))
    
    # extract_mask: push float theta + tau + box... (48 bytes)
    pc_extract = [PushConstantRange(SHADER_STAGE_COMPUTE_BIT, 0, 48)]
    state.layout_extract_mask = unwrap(create_pipeline_layout(dev, PipelineLayoutCreateInfo([state.dsl_extract_mask], pc_extract)))
    
    # diffusivity: push float epsilon_D + float gamma (8 bytes)
    pc_diff = [PushConstantRange(SHADER_STAGE_COMPUTE_BIT, 0, 8)]
    state.layout_diffusivity = unwrap(create_pipeline_layout(dev, PipelineLayoutCreateInfo([state.dsl_diffusivity], pc_diff)))
    
    # slice: push float theta + int slice_dim + int slice_idx + uint active_id (16 bytes)
    pc_slice = [PushConstantRange(SHADER_STAGE_COMPUTE_BIT, 0, 16)]
    state.layout_slice = unwrap(create_pipeline_layout(dev, PipelineLayoutCreateInfo([state.dsl_slice], pc_slice)))
    
    # 5. Create compute pipelines
    stage_seed = PipelineShaderStageCreateInfo(SHADER_STAGE_COMPUTE_BIT, mod_seed, "main")
    state.pipeline_seed_init = unwrap(create_compute_pipelines(dev, [ComputePipelineCreateInfo(stage_seed, state.layout_seed_init, -1)]))[1][1]
    
    stage_diffuse = PipelineShaderStageCreateInfo(SHADER_STAGE_COMPUTE_BIT, mod_diffuse, "main")
    state.pipeline_diffuse_step = unwrap(create_compute_pipelines(dev, [ComputePipelineCreateInfo(stage_diffuse, state.layout_diffuse_step, -1)]))[1][1]
    
    stage_extract = PipelineShaderStageCreateInfo(SHADER_STAGE_COMPUTE_BIT, mod_extract, "main")
    state.pipeline_extract_mask = unwrap(create_compute_pipelines(dev, [ComputePipelineCreateInfo(stage_extract, state.layout_extract_mask, -1)]))[1][1]
    
    stage_diff = PipelineShaderStageCreateInfo(SHADER_STAGE_COMPUTE_BIT, mod_diff, "main")
    state.pipeline_diffusivity = unwrap(create_compute_pipelines(dev, [ComputePipelineCreateInfo(stage_diff, state.layout_diffusivity, -1)]))[1][1]

    stage_slice = PipelineShaderStageCreateInfo(SHADER_STAGE_COMPUTE_BIT, mod_slice, "main")
    state.pipeline_slice = unwrap(create_compute_pipelines(dev, [ComputePipelineCreateInfo(stage_slice, state.layout_slice, -1)]))[1][1]
    
    # 6. Shader modules are garbage collected automatically by Vulkan.jl
    # destroy_shader_module(dev, mod_seed)
    # destroy_shader_module(dev, mod_diffuse)
    # destroy_shader_module(dev, mod_extract)
    # destroy_shader_module(dev, mod_diff)
    # destroy_shader_module(dev, mod_slice)
    
    # 7. Create descriptor pool and allocate descriptor sets
    pool_sizes = [
        DescriptorPoolSize(DESCRIPTOR_TYPE_STORAGE_IMAGE, 64)  # plenty for all sets
    ]
    state.descriptor_pool = unwrap(create_descriptor_pool(dev, DescriptorPoolCreateInfo(32, pool_sizes)))
    
    # Create dummy sampler for DescriptorImageInfo
    sampler_info = SamplerCreateInfo(
        FILTER_NEAREST, FILTER_NEAREST, SAMPLER_MIPMAP_MODE_NEAREST,
        SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        0.0f0, false, 1.0f0, false, COMPARE_OP_NEVER, 0.0f0, 0.0f0, BORDER_COLOR_FLOAT_TRANSPARENT_BLACK, false
    )
    state.dummy_sampler = unwrap(create_sampler(dev, sampler_info))
    
    # Allocate all descriptor sets
    alloc_ds = DescriptorSetAllocateInfo(state.descriptor_pool, [
        state.dsl_seed_init,       # ds_seed_ping
        state.dsl_seed_init,       # ds_seed_pong
        state.dsl_diffuse_step,    # ds_diffuse_ping_to_pong
        state.dsl_diffuse_step,    # ds_diffuse_pong_to_ping
        state.dsl_extract_mask,    # ds_extract_from_ping
        state.dsl_extract_mask,    # ds_extract_from_pong
        state.dsl_diffusivity,     # ds_compute_diffusivity
        state.dsl_slice,           # ds_slice_ping
        state.dsl_slice            # ds_slice_pong
    ])
    sets = unwrap(Vulkan.allocate_descriptor_sets(dev, alloc_ds))
    state.ds_seed_ping = sets[1]
    state.ds_seed_pong = sets[2]
    state.ds_diffuse_ping_to_pong = sets[3]
    state.ds_diffuse_pong_to_ping = sets[4]
    state.ds_extract_from_ping = sets[5]
    state.ds_extract_from_pong = sets[6]
    state.ds_compute_diffusivity = sets[7]
    state.ds_slice_ping = sets[8]
    state.ds_slice_pong = sets[9]
    
    # 8. Write descriptor sets
    _write_descriptors!(state, dev)
    
    # 9. Create compute command pool & buffer
    pool_info = CommandPoolCreateInfo(ctx.queue_family_index; flags=COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT)
    state.compute_cmd_pool = unwrap(create_command_pool(dev, pool_info))
    
    alloc_cmd = CommandBufferAllocateInfo(state.compute_cmd_pool, COMMAND_BUFFER_LEVEL_PRIMARY, 1)
    state.compute_cmd_buffer = unwrap(Vulkan.allocate_command_buffers(dev, alloc_cmd))[1]
    
    state.compute_fence = unwrap(create_fence(dev, FenceCreateInfo()))
    
    # Allocate 16 MB persistent staging box buffer
    box_buf_size = 16 * 1024 * 1024
    buf_info = BufferCreateInfo(box_buf_size, BUFFER_USAGE_TRANSFER_DST_BIT, SHARING_MODE_EXCLUSIVE, UInt32[])
    state.staging_box_buf = unwrap(create_buffer(dev, buf_info))
    
    mem_reqs = get_buffer_memory_requirements(dev, state.staging_box_buf)
    mem_props = get_physical_device_memory_properties(pdev)
    mem_type = UInt32(0)
    found_type = false
    for pass in 1:2
        target_flags = MEMORY_PROPERTY_HOST_VISIBLE_BIT | MEMORY_PROPERTY_HOST_COHERENT_BIT
        if pass == 1
            target_flags |= MEMORY_PROPERTY_HOST_CACHED_BIT
        end
        
        for i in 0:(mem_props.memory_type_count - 1)
            if (mem_reqs.memory_type_bits & (1 << i)) != 0 &&
               (mem_props.memory_types[i + 1].property_flags & target_flags) == target_flags
                try
                    state.staging_box_mem = unwrap(allocate_memory(dev, MemoryAllocateInfo(mem_reqs.size, UInt32(i))))
                    mem_type = UInt32(i)
                    found_type = true
                    break
                catch e
                    if e isa Vulkan.VulkanError && e.code == Vulkan.ERROR_OUT_OF_DEVICE_MEMORY
                        @warn "Staging buffer memory allocation failed on heap $(i), trying next suitable heap..."
                        continue
                    end
                    rethrow(e)
                end
            end
        end
        if found_type; break; end
    end
    if !found_type
        error("Failed to find HOST_VISIBLE | HOST_COHERENT memory or out of memory for staging box buffer")
    end
    unwrap(bind_buffer_memory(dev, state.staging_box_buf, state.staging_box_mem, 0))
    state.staging_box_ptr = unwrap(map_memory(dev, state.staging_box_mem, 0, box_buf_size))
    state.staging_box_capacity = box_buf_size

    state.is_initialized = true
    @info "[HeatDiffusion] Initialization complete (allocated 16 MB persistent staging buffer)."
    return state
end

function _write_descriptors!(state::HeatDiffusionState, dev::Device)
    # Helper to create a DescriptorImageInfo for a storage image
    di(view) = DescriptorImageInfo(state.dummy_sampler, view, IMAGE_LAYOUT_GENERAL)
    
    writes = WriteDescriptorSet[]
    
    # seed_init → ping
    push!(writes, WriteDescriptorSet(state.ds_seed_ping, 0, 0, DESCRIPTOR_TYPE_STORAGE_IMAGE, [di(state.view_u_ping)], [], []))
    # seed_init → pong
    push!(writes, WriteDescriptorSet(state.ds_seed_pong, 0, 0, DESCRIPTOR_TYPE_STORAGE_IMAGE, [di(state.view_u_pong)], [], []))
    
    # diffuse: ping→pong (D, ping_in, pong_out)
    push!(writes, WriteDescriptorSet(state.ds_diffuse_ping_to_pong, 0, 0, DESCRIPTOR_TYPE_STORAGE_IMAGE, [di(state.view_diffusivity)], [], []))
    push!(writes, WriteDescriptorSet(state.ds_diffuse_ping_to_pong, 1, 0, DESCRIPTOR_TYPE_STORAGE_IMAGE, [di(state.view_u_ping)], [], []))
    push!(writes, WriteDescriptorSet(state.ds_diffuse_ping_to_pong, 2, 0, DESCRIPTOR_TYPE_STORAGE_IMAGE, [di(state.view_u_pong)], [], []))
    
    # diffuse: pong→ping (D, pong_in, ping_out)
    push!(writes, WriteDescriptorSet(state.ds_diffuse_pong_to_ping, 0, 0, DESCRIPTOR_TYPE_STORAGE_IMAGE, [di(state.view_diffusivity)], [], []))
    push!(writes, WriteDescriptorSet(state.ds_diffuse_pong_to_ping, 1, 0, DESCRIPTOR_TYPE_STORAGE_IMAGE, [di(state.view_u_pong)], [], []))
    push!(writes, WriteDescriptorSet(state.ds_diffuse_pong_to_ping, 2, 0, DESCRIPTOR_TYPE_STORAGE_IMAGE, [di(state.view_u_ping)], [], []))
    
    # extract from ping (u_ping, mask_out)
    push!(writes, WriteDescriptorSet(state.ds_extract_from_ping, 0, 0, DESCRIPTOR_TYPE_STORAGE_IMAGE, [di(state.view_u_ping)], [], []))
    push!(writes, WriteDescriptorSet(state.ds_extract_from_ping, 1, 0, DESCRIPTOR_TYPE_STORAGE_IMAGE, [di(state.view_mask)], [], []))
    
    # extract from pong (u_pong, mask_out)
    push!(writes, WriteDescriptorSet(state.ds_extract_from_pong, 0, 0, DESCRIPTOR_TYPE_STORAGE_IMAGE, [di(state.view_u_pong)], [], []))
    push!(writes, WriteDescriptorSet(state.ds_extract_from_pong, 1, 0, DESCRIPTOR_TYPE_STORAGE_IMAGE, [di(state.view_mask)], [], []))
    
    # diffusivity compute: edge_in → diffusivity_out
    push!(writes, WriteDescriptorSet(state.ds_compute_diffusivity, 0, 0, DESCRIPTOR_TYPE_STORAGE_IMAGE, [di(state.view_u_ping)], [], []))  # reuse ping for edge input
    push!(writes, WriteDescriptorSet(state.ds_compute_diffusivity, 1, 0, DESCRIPTOR_TYPE_STORAGE_IMAGE, [di(state.view_diffusivity)], [], []))
    
    # slice to texture: u_ping/u_pong → tex_out
    push!(writes, WriteDescriptorSet(state.ds_slice_ping, 0, 0, DESCRIPTOR_TYPE_STORAGE_IMAGE, [di(state.view_u_ping)], [], []))
    push!(writes, WriteDescriptorSet(state.ds_slice_pong, 0, 0, DESCRIPTOR_TYPE_STORAGE_IMAGE, [di(state.view_u_pong)], [], []))
    # Note: binding 1 (tex_out) is written dynamically per-tick since the texture view changes
    
    update_descriptor_sets(dev, writes, [])
end

# ─── Upload diffusivity field ────────────────────────────────────────────

"""
    upload_diffusivity!(state, ctx, diffusivity_data::Array{Float32, 3})

Uploads the precomputed diffusivity field D(x) to the GPU.
`diffusivity_data` must be size (w, h, d).
"""
function upload_diffusivity!(state::HeatDiffusionState, ctx, diffusivity_data::Array{Float32, 3})
    @assert state.is_initialized "Heat diffusion not initialized"
    @assert size(diffusivity_data) == (state.width, state.height, state.depth) "Diffusivity size mismatch"
    
    dev = ctx.device
    pdev = ctx.physical_device
    w, h, d = state.width, state.height, state.depth
    data_size = sizeof(diffusivity_data)
    
    @info "[HeatDiffusion] Uploading diffusivity field ($(w)×$(h)×$(d), $(data_size ÷ 1024) KB)..."
    
    # Create staging buffer
    buf_info = BufferCreateInfo(data_size, BUFFER_USAGE_TRANSFER_SRC_BIT, SHARING_MODE_EXCLUSIVE, UInt32[])
    staging_buf = unwrap(create_buffer(dev, buf_info))
    
    mem_reqs = get_buffer_memory_requirements(dev, staging_buf)
    mem_props = get_physical_device_memory_properties(pdev)
    mem_type = UInt32(0)
    found_type = false
    local staging_mem
    for i in 0:(mem_props.memory_type_count - 1)
        if (mem_reqs.memory_type_bits & (1 << i)) != 0 &&
           (mem_props.memory_types[i + 1].property_flags & (MEMORY_PROPERTY_HOST_VISIBLE_BIT | MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
           (MEMORY_PROPERTY_HOST_VISIBLE_BIT | MEMORY_PROPERTY_HOST_COHERENT_BIT)
            try
                staging_mem = unwrap(allocate_memory(dev, MemoryAllocateInfo(mem_reqs.size, UInt32(i))))
                mem_type = UInt32(i)
                found_type = true
                break
            catch e
                if e isa Vulkan.VulkanError && e.code == Vulkan.ERROR_OUT_OF_DEVICE_MEMORY
                    @warn "Diffusivity staging buffer memory allocation failed on heap $(i), trying next suitable heap..."
                    continue
                end
                rethrow(e)
            end
        end
    end
    if !found_type
        error("Failed to find suitable memory for diffusivity staging buffer or out of memory")
    end
    unwrap(bind_buffer_memory(dev, staging_buf, staging_mem, 0))
    
    # Map and copy data
    ptr = unwrap(map_memory(dev, staging_mem, 0, data_size))
    unsafe_copyto!(Ptr{Float32}(ptr), pointer(diffusivity_data), length(diffusivity_data))
    unmap_memory(dev, staging_mem)
    
    # Transition image layout and copy
    one_time_submit!(ctx) do cmd
        # Transition diffusivity image to TRANSFER_DST
        barrier = ImageMemoryBarrier(
            ACCESS_NONE, ACCESS_TRANSFER_WRITE_BIT,
            IMAGE_LAYOUT_UNDEFINED, IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            0, 0,  # queue family indices (ignored)
            state.img_diffusivity,
            ImageSubresourceRange(IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1)
        )
        cmd_pipeline_barrier(cmd, MemoryBarrier[], BufferMemoryBarrier[], [barrier];
                             src_stage_mask=PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             dst_stage_mask=PIPELINE_STAGE_TRANSFER_BIT)
        
        # Copy buffer to 3D image
        region = BufferImageCopy(
            0, 0, 0,
            ImageSubresourceLayers(IMAGE_ASPECT_COLOR_BIT, 0, 0, 1),
            Offset3D(0, 0, 0),
            Extent3D(w, h, d)
        )
        cmd_copy_buffer_to_image(cmd, staging_buf, state.img_diffusivity, IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, [region])
        
        # Transition to GENERAL for compute shader access
        barrier2 = ImageMemoryBarrier(
            ACCESS_TRANSFER_WRITE_BIT, ACCESS_SHADER_READ_BIT,
            IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, IMAGE_LAYOUT_GENERAL,
            0, 0,
            state.img_diffusivity,
            ImageSubresourceRange(IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1)
        )
        cmd_pipeline_barrier(cmd, MemoryBarrier[], BufferMemoryBarrier[], [barrier2];
                             src_stage_mask=PIPELINE_STAGE_TRANSFER_BIT,
                             dst_stage_mask=PIPELINE_STAGE_COMPUTE_SHADER_BIT)
    end
    
    # Staging buffer and memory will be cleaned up by Vulkan.jl GC finalizers
    staging_buf = nothing
    staging_mem = nothing
    GC.gc(false)
    
    state.diffusivity_uploaded = true
    @info "[HeatDiffusion] Diffusivity upload complete."
end

# ─── Run heat diffusion ─────────────────────────────────────────────────

"""
    run_heat_diffusion!(state, ctx, seed_x, seed_y, seed_z; K=200, dt=0.16f0, theta=0.001044f0, tau=0.0001f0)

Runs the complete heat diffusion pipeline on the GPU:
1. Initialize heat field with seed impulse
2. Run K steps of diffusion
3. Extract binary mask

Returns nothing — the mask is stored in `state.img_mask` on the GPU.
Use `read_mask_to_cpu` to retrieve it.
"""
function run_heat_diffusion!(state::HeatDiffusionState, ctx, seed_x::Int, seed_y::Int, seed_z::Int;
                              K::Int=200, dt::Float32=0.16f0, theta::Float32=0.001044f0, tau::Float32=0.0001f0,
                              spacing::NTuple{3, Real}=state.spacing, s_boost::Float32=1.0f0,
                              edge_params::Union{NTuple{4, Real}, NTuple{10, Real}, NTuple{12, Real}, AbstractVector{<:Real}}=state.current_edge_params)
    @assert state.is_initialized "Heat diffusion not initialized"
    @assert state.diffusivity_uploaded "Diffusivity not uploaded"
    
    state.spacing = (Float64(spacing[1]), Float64(spacing[2]), Float64(spacing[3]))
    state.current_s_boost = s_boost
    state.current_edge_params = _normalize_edge_params(edge_params)
    inv_dx2, inv_dy2, inv_dz2 = _compute_spacing_weights(state.spacing)
    
    dev = ctx.device
    w, h, d = state.width, state.height, state.depth
    
    # Compute dispatch dimensions (ceil division by workgroup size 8)
    gx = cld(w, 8)
    gy = cld(h, 8)
    gz = cld(d, 8)
    
    # Record command buffer
    cmd = state.compute_cmd_buffer
    unwrap(reset_command_buffer(cmd))
    begin_info = CommandBufferBeginInfo(flags = COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT)
    unwrap(begin_command_buffer(cmd, begin_info))
    
    # --- Step 1: Transition ping/pong images to GENERAL ---
    barriers_init = [
        ImageMemoryBarrier(C_NULL, ACCESS_NONE, ACCESS_SHADER_WRITE_BIT,
            IMAGE_LAYOUT_UNDEFINED, IMAGE_LAYOUT_GENERAL, 0, 0,
            state.img_u_ping, ImageSubresourceRange(IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1)),
        ImageMemoryBarrier(C_NULL, ACCESS_NONE, ACCESS_SHADER_WRITE_BIT,
            IMAGE_LAYOUT_UNDEFINED, IMAGE_LAYOUT_GENERAL, 0, 0,
            state.img_u_pong, ImageSubresourceRange(IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1)),
        ImageMemoryBarrier(C_NULL, ACCESS_NONE, ACCESS_SHADER_WRITE_BIT,
            IMAGE_LAYOUT_UNDEFINED, IMAGE_LAYOUT_GENERAL, 0, 0,
            state.img_mask, ImageSubresourceRange(IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1))
    ]
    cmd_pipeline_barrier(cmd, MemoryBarrier[], BufferMemoryBarrier[], barriers_init;
                         src_stage_mask=PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         dst_stage_mask=PIPELINE_STAGE_COMPUTE_SHADER_BIT)
    
    # --- Step 2: Initialize seed in ping buffer ---
    cmd_bind_pipeline(cmd, PIPELINE_BIND_POINT_COMPUTE, state.pipeline_seed_init)
    cmd_bind_descriptor_sets(cmd, PIPELINE_BIND_POINT_COMPUTE, state.layout_seed_init, 0, [state.ds_seed_ping], [])
    
    pc_seed = SeedInitPC(
        Int32(seed_x - 1), Int32(seed_y - 1), Int32(seed_z - 1), Int32(0),
        Int32(0), Int32(0), Int32(0), Int32(0),
        inv_dx2, inv_dy2, inv_dz2, 0.0f0
    )
    ref_seed = Ref(pc_seed)
    GC.@preserve ref_seed begin
        cmd_push_constants(cmd, state.layout_seed_init, SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc_seed), Base.unsafe_convert(Ptr{Cvoid}, ref_seed))
    end
    cmd_dispatch(cmd, gx, gy, gz)
    
    # Barrier: seed write → diffuse read
    barrier_compute = MemoryBarrier(C_NULL, ACCESS_SHADER_WRITE_BIT, ACCESS_SHADER_READ_BIT)
    cmd_pipeline_barrier(cmd, [barrier_compute], BufferMemoryBarrier[], ImageMemoryBarrier[];
                         src_stage_mask=PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         dst_stage_mask=PIPELINE_STAGE_COMPUTE_SHADER_BIT)
    
    # --- Step 3: Run K diffusion steps (ping-pong) ---
    cmd_bind_pipeline(cmd, PIPELINE_BIND_POINT_COMPUTE, state.pipeline_diffuse_step)
    
    ep = state.current_edge_params
    pc_diffuse = DiffuseStepPC(
        dt, Int32(0), Int32(seed_x - 1), Int32(seed_y - 1),
        Int32(0), Int32(0), Int32(0), Int32(seed_z - 1),
        Int32(0), Int32(0), Int32(0), Int32(3), # default seed radius 3
        inv_dx2, inv_dy2, inv_dz2, s_boost,
        ep[1], ep[2], ep[3], ep[4], ep[5], ep[6],
        ep[7], ep[8], ep[9], ep[10], ep[11], ep[12]
    )
    
    for k in 1:K
        if isodd(k)
            # Read from ping, write to pong
            cmd_bind_descriptor_sets(cmd, PIPELINE_BIND_POINT_COMPUTE, state.layout_diffuse_step, 0, [state.ds_diffuse_ping_to_pong], [])
        else
            # Read from pong, write to ping
            cmd_bind_descriptor_sets(cmd, PIPELINE_BIND_POINT_COMPUTE, state.layout_diffuse_step, 0, [state.ds_diffuse_pong_to_ping], [])
        end
        ref_diffuse = Ref(pc_diffuse)
        GC.@preserve ref_diffuse begin
            cmd_push_constants(cmd, state.layout_diffuse_step, SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc_diffuse), Base.unsafe_convert(Ptr{Cvoid}, ref_diffuse))
        end
        cmd_dispatch(cmd, gx, gy, gz)
        
        # Memory barrier between steps
        cmd_pipeline_barrier(cmd, [barrier_compute], BufferMemoryBarrier[], ImageMemoryBarrier[];
                             src_stage_mask=PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             dst_stage_mask=PIPELINE_STAGE_COMPUTE_SHADER_BIT)
    end
    
    # --- Step 4: Extract mask ---
    cmd_bind_pipeline(cmd, PIPELINE_BIND_POINT_COMPUTE, state.pipeline_extract_mask)
    
    # Determine which buffer has the final result (depends on K parity)
    if isodd(K)
        cmd_bind_descriptor_sets(cmd, PIPELINE_BIND_POINT_COMPUTE, state.layout_extract_mask, 0, [state.ds_extract_from_pong], [])
    else
        cmd_bind_descriptor_sets(cmd, PIPELINE_BIND_POINT_COMPUTE, state.layout_extract_mask, 0, [state.ds_extract_from_ping], [])
    end
    
    # use_box=0 since run_full_diffusion! does the whole volume, but we must pass all 12 arguments to the struct constructor
    pc_extract = ExtractMaskPC(Float32(theta), Float32(tau), Int32(1), Int32(0), Int32(0), Int32(0), Int32(0), Int32(0), Int32(0), Int32(0), Int32(0), Int32(0))
    ref_extract = Ref(pc_extract)
    GC.@preserve ref_extract begin
        cmd_push_constants(cmd, state.layout_extract_mask, SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc_extract), Base.unsafe_convert(Ptr{Cvoid}, ref_extract))
    end
    cmd_dispatch(cmd, gx, gy, gz)
    
    # Final barrier
    cmd_pipeline_barrier(cmd, [barrier_compute], BufferMemoryBarrier[], ImageMemoryBarrier[];
                         src_stage_mask=PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         dst_stage_mask=PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT)
    
    unwrap(end_command_buffer(cmd))
    
    # Submit and wait
    submit_info = SubmitInfo([], [], [cmd], [])
    unwrap(reset_fences(dev, [state.compute_fence]))
    unwrap(queue_submit(ctx.graphics_queue, [submit_info]; fence=state.compute_fence))
    unwrap(wait_for_fences(dev, [state.compute_fence], true, typemax(UInt64)))
end

# ─── Read mask back to CPU ───────────────────────────────────────────────

"""
    read_mask_to_cpu(state, ctx) → Array{Float32, 3}

Reads the computed mask from GPU back to CPU memory.
"""
function read_mask_to_cpu(state::HeatDiffusionState, ctx)::Array{Float32, 3}
    @assert state.is_initialized "Heat diffusion not initialized"
    
    dev = ctx.device
    pdev = ctx.physical_device
    w, h, d = state.width, state.height, state.depth
    data_size = w * h * d * sizeof(Float32)
    
    # Create staging buffer for readback
    buf_info = BufferCreateInfo(data_size, BUFFER_USAGE_TRANSFER_DST_BIT, SHARING_MODE_EXCLUSIVE, UInt32[])
    staging_buf = unwrap(create_buffer(dev, buf_info))
    
    mem_reqs = get_buffer_memory_requirements(dev, staging_buf)
    mem_props = get_physical_device_memory_properties(pdev)
    mem_type = UInt32(0)
    found_type = false
    local staging_mem
    for i in 0:(mem_props.memory_type_count - 1)
        if (mem_reqs.memory_type_bits & (1 << i)) != 0 &&
           (mem_props.memory_types[i + 1].property_flags & (MEMORY_PROPERTY_HOST_VISIBLE_BIT | MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
           (MEMORY_PROPERTY_HOST_VISIBLE_BIT | MEMORY_PROPERTY_HOST_COHERENT_BIT)
            try
                staging_mem = unwrap(allocate_memory(dev, MemoryAllocateInfo(mem_reqs.size, UInt32(i))))
                mem_type = UInt32(i)
                found_type = true
                break
            catch e
                if e isa Vulkan.VulkanError && e.code == Vulkan.ERROR_OUT_OF_DEVICE_MEMORY
                    @warn "Mask staging buffer memory allocation failed on heap $(i), trying next suitable heap..."
                    continue
                end
                rethrow(e)
            end
        end
    end
    if !found_type
        error("Failed to find suitable memory for mask staging buffer or out of memory")
    end
    unwrap(bind_buffer_memory(dev, staging_buf, staging_mem, 0))
    
    # Copy image to buffer
    one_time_submit!(ctx) do cmd
        barrier = ImageMemoryBarrier(
            ACCESS_SHADER_WRITE_BIT, ACCESS_TRANSFER_READ_BIT,
            IMAGE_LAYOUT_GENERAL, IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            0, 0,
            state.img_mask,
            ImageSubresourceRange(IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1)
        )
        cmd_pipeline_barrier(cmd, MemoryBarrier[], BufferMemoryBarrier[], [barrier];
                             src_stage_mask=PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             dst_stage_mask=PIPELINE_STAGE_TRANSFER_BIT)
        
        region = BufferImageCopy(
            0, 0, 0,
            ImageSubresourceLayers(IMAGE_ASPECT_COLOR_BIT, 0, 0, 1),
            Offset3D(0, 0, 0),
            Extent3D(w, h, d)
        )
        cmd_copy_image_to_buffer(cmd, state.img_mask, IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, staging_buf, [region])
        
        # Transition back to GENERAL
        barrier2 = ImageMemoryBarrier(
            ACCESS_TRANSFER_READ_BIT, ACCESS_SHADER_WRITE_BIT,
            IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, IMAGE_LAYOUT_GENERAL,
            0, 0,
            state.img_mask,
            ImageSubresourceRange(IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1)
        )
        cmd_pipeline_barrier(cmd, MemoryBarrier[], BufferMemoryBarrier[], [barrier2];
                             src_stage_mask=PIPELINE_STAGE_TRANSFER_BIT,
                             dst_stage_mask=PIPELINE_STAGE_COMPUTE_SHADER_BIT)
    end
    
    # Map and read data
    ptr = unwrap(map_memory(dev, staging_mem, 0, data_size))
    result = Array{Float32, 3}(undef, w, h, d)
    unsafe_copyto!(pointer(result), Ptr{Float32}(ptr), length(result))
    unmap_memory(dev, staging_mem)
    
    # Staging buffer and memory will be cleaned up by Vulkan.jl GC finalizers
    staging_buf = nothing
    staging_mem = nothing
    GC.gc(false)
    
    return result
end

# ─── Incremental (interactive) diffusion ─────────────────────────────────

"""
    seed_only!(state, ctx, seed_x, seed_y, seed_z; seed_radius=3, spacing=state.spacing, s_boost=1.0f0)

Initialize the heat field with a single seed point at (seed_x, seed_y, seed_z)
without running any diffusion steps. Call `run_incremental_steps!` afterwards
to add diffusion steps incrementally (e.g., per-frame during a mouse hold).
"""
function seed_only!(state::HeatDiffusionState, ctx, seed_x::Int, seed_y::Int, seed_z::Int;
                     seed_radius::Int=3, spacing::NTuple{3, Real}=state.spacing, s_boost::Float32=1.0f0,
                     edge_params::Union{NTuple{4, Real}, NTuple{10, Real}, NTuple{12, Real}, AbstractVector{<:Real}}=state.current_edge_params)
    @assert state.is_initialized "Heat diffusion not initialized"
    @assert state.diffusivity_uploaded "Diffusivity not uploaded"
    
    state.spacing = (Float64(spacing[1]), Float64(spacing[2]), Float64(spacing[3]))
    state.current_s_boost = s_boost
    state.current_edge_params = _normalize_edge_params(edge_params)
    inv_dx2, inv_dy2, inv_dz2 = _compute_spacing_weights(state.spacing)
    
    dev = ctx.device
    w, h, d = state.width, state.height, state.depth
    gx, gy, gz = cld(w, 8), cld(h, 8), cld(d, 8)
    
    cmd = state.compute_cmd_buffer
    unwrap(reset_command_buffer(cmd))
    begin_info = CommandBufferBeginInfo(flags = COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT)
    unwrap(begin_command_buffer(cmd, begin_info))
    
    # Transition ping, pong, mask images to GENERAL layout
    barriers_init = [
        ImageMemoryBarrier(C_NULL, ACCESS_NONE, ACCESS_SHADER_WRITE_BIT,
            IMAGE_LAYOUT_UNDEFINED, IMAGE_LAYOUT_GENERAL, 0, 0,
            state.img_u_ping, ImageSubresourceRange(IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1)),
        ImageMemoryBarrier(C_NULL, ACCESS_NONE, ACCESS_SHADER_WRITE_BIT,
            IMAGE_LAYOUT_UNDEFINED, IMAGE_LAYOUT_GENERAL, 0, 0,
            state.img_u_pong, ImageSubresourceRange(IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1)),
        ImageMemoryBarrier(C_NULL, ACCESS_NONE, ACCESS_SHADER_WRITE_BIT,
            IMAGE_LAYOUT_UNDEFINED, IMAGE_LAYOUT_GENERAL, 0, 0,
            state.img_mask, ImageSubresourceRange(IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1))
    ]
    cmd_pipeline_barrier(cmd, MemoryBarrier[], BufferMemoryBarrier[], barriers_init;
                         src_stage_mask=PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         dst_stage_mask=PIPELINE_STAGE_COMPUTE_SHADER_BIT)
    
    # Clear ping, pong, and mask buffers to 0.0 (fast fixed-function clear)
    clear_color = ClearColorValue((0.0f0, 0.0f0, 0.0f0, 0.0f0))
    subres = ImageSubresourceRange(IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1)
    cmd_clear_color_image(cmd, state.img_u_ping, IMAGE_LAYOUT_GENERAL, clear_color, [subres])
    cmd_clear_color_image(cmd, state.img_u_pong, IMAGE_LAYOUT_GENERAL, clear_color, [subres])
    cmd_clear_color_image(cmd, state.img_mask, IMAGE_LAYOUT_GENERAL, clear_color, [subres])
    
    # Barrier after clear
    clear_barrier = MemoryBarrier(C_NULL, ACCESS_TRANSFER_WRITE_BIT, ACCESS_SHADER_WRITE_BIT)
    cmd_pipeline_barrier(cmd, [clear_barrier], BufferMemoryBarrier[], ImageMemoryBarrier[];
                         src_stage_mask=PIPELINE_STAGE_TRANSFER_BIT,
                         dst_stage_mask=PIPELINE_STAGE_COMPUTE_SHADER_BIT)
    
    # Seed init into ping buffer
    cmd_bind_pipeline(cmd, PIPELINE_BIND_POINT_COMPUTE, state.pipeline_seed_init)
    cmd_bind_descriptor_sets(cmd, PIPELINE_BIND_POINT_COMPUTE, state.layout_seed_init, 0, [state.ds_seed_ping], [])
    
    # Calculate small bounding box around seed
    r = seed_radius
    xmin = max(1, seed_x - r)
    xmax = min(w, seed_x + r)
    ymin = max(1, seed_y - r)
    ymax = min(h, seed_y + r)
    zmin = max(1, seed_z - r)
    zmax = min(d, seed_z + r)
    
    bw = xmax - xmin + 1
    bh = ymax - ymin + 1
    bd = zmax - zmin + 1
    
    pc_seed = SeedInitPC(
        Int32(seed_x - 1), Int32(seed_y - 1), Int32(seed_z - 1), Int32(seed_radius),
        Int32(xmin - 1), Int32(ymin - 1), Int32(zmin - 1), Int32(0),
        inv_dx2, inv_dy2, inv_dz2, 0.0f0
    )
    ref_seed = Ref(pc_seed)
    GC.@preserve ref_seed begin
        cmd_push_constants(cmd, state.layout_seed_init, SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc_seed), Base.unsafe_convert(Ptr{Cvoid}, ref_seed))
    end
    cmd_dispatch(cmd, cld(bw, 8), cld(bh, 8), cld(bd, 8))
    
    barrier_compute = MemoryBarrier(C_NULL, ACCESS_SHADER_WRITE_BIT, ACCESS_SHADER_READ_BIT)
    cmd_pipeline_barrier(cmd, [barrier_compute], BufferMemoryBarrier[], ImageMemoryBarrier[];
                         src_stage_mask=PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         dst_stage_mask=PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT)
    
    unwrap(end_command_buffer(cmd))
    submit_info = SubmitInfo([], [], [cmd], [])
    unwrap(reset_fences(dev, [state.compute_fence]))
    unwrap(queue_submit(ctx.graphics_queue, [submit_info]; fence=state.compute_fence))
    unwrap(wait_for_fences(dev, [state.compute_fence], true, typemax(UInt64)))
    
    state.current_buffer_is_ping = true
    state.accumulated_steps = 0
    state.seed_x = seed_x
    state.seed_y = seed_y
    state.seed_z = seed_z
    state.seed_radius = seed_radius
    @debug "[HeatDiffusion] Seeded at ($seed_x, $seed_y, $seed_z)"
end

"""
    run_incremental_steps!(state, ctx, n_steps; dt=0.16f0, theta=0.001f0, tau=0.0001f0, copy_box_to_cpu=false, spacing=state.spacing)

Runs `n_steps` additional diffusion steps on the GPU without re-seeding,
continuing from whichever ping/pong buffer currently holds the latest data.
After the diffusion steps, automatically runs the extract_mask shader so
`read_mask_to_cpu` can be called immediately.
"""
function run_incremental_steps!(state::HeatDiffusionState, ctx, n_steps::Int;
                                 dt::Float32=0.16f0, theta::Float32=0.001f0, tau::Float32=0.0001f0, copy_box_to_cpu::Bool=false,
                                 spacing::NTuple{3, Real}=state.spacing, s_boost::Float32=state.current_s_boost,
                                 edge_params::Union{NTuple{4, Real}, NTuple{10, Real}, NTuple{12, Real}, AbstractVector{<:Real}}=state.current_edge_params)
    @assert state.is_initialized "Heat diffusion not initialized"
    @assert state.diffusivity_uploaded "Diffusivity not uploaded"
    n_steps <= 0 && !copy_box_to_cpu && return
    
    state.spacing = (Float64(spacing[1]), Float64(spacing[2]), Float64(spacing[3]))
    state.current_s_boost = s_boost
    state.current_edge_params = _normalize_edge_params(edge_params)
    inv_dx2, inv_dy2, inv_dz2 = _compute_spacing_weights(state.spacing)
    
    dev = ctx.device
    w, h, d = state.width, state.height, state.depth
    
    total_steps = state.accumulated_steps + n_steps
    # Expand bounding box based on actual heat diffusion distance (approx 3 std devs)
    # std_dev = sqrt(2 * D * t) where D <= 1.0 and t = total_steps * dt
    # 3 * sqrt(2 * 0.16 * total_steps) ≈ 3 * sqrt(0.32 * total_steps) ≈ 1.7 * sqrt(total_steps)
    R = state.seed_radius + round(Int, 2.0 * sqrt(max(0, total_steps))) + 6
    xmin = max(1, state.seed_x - R)
    xmax = min(w, state.seed_x + R)
    ymin = max(1, state.seed_y - R)
    ymax = min(h, state.seed_y + R)
    zmin = max(1, state.seed_z - R)
    zmax = min(d, state.seed_z + R)
    
    # Cap bounding box to fit in 16MB buffer if needed
    bw = xmax - xmin + 1
    bh = ymax - ymin + 1
    bd = zmax - zmin + 1
    if (bw * bh * bd * 4) > state.staging_box_capacity
        bw = bh = bd = round(Int, cbrt(state.staging_box_capacity / 4)) - 1
        xmax = min(w, xmin + bw - 1)
        ymax = min(h, ymin + bh - 1)
        zmax = min(d, zmin + bd - 1)
        bw = xmax - xmin + 1
        bh = ymax - ymin + 1
        bd = zmax - zmin + 1
    end
    
    state.current_box_min = (xmin, ymin, zmin)
    state.current_box_max = (xmax, ymax, zmax)
    
    gx = cld(bw, 8)
    gy = cld(bh, 8)
    gz = cld(bd, 8)
    
    # DEBUG
    # println("[HeatDiffusion] total_steps=$total_steps, R=$R, bbox=($bw x $bh x $bd), workgroups=($gx x $gy x $gz)")
    
    cmd = state.compute_cmd_buffer
    unwrap(reset_command_buffer(cmd))
    begin_info = CommandBufferBeginInfo(flags = COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT)
    unwrap(begin_command_buffer(cmd, begin_info))
    
    barrier_compute = MemoryBarrier(C_NULL, ACCESS_SHADER_WRITE_BIT, ACCESS_SHADER_READ_BIT)
    
    # Push constants for bounding box
    box_min_glsl = Int32[xmin - 1, ymin - 1, zmin - 1, 0]
    box_max_glsl = Int32[xmax - 1, ymax - 1, zmax - 1, 0]
    
    ep = state.current_edge_params
    pc_diffuse = DiffuseStepPC(
        dt, Int32(1), Int32(state.seed_x - 1), Int32(state.seed_y - 1),
        box_min_glsl[1], box_min_glsl[2], box_min_glsl[3], Int32(state.seed_z - 1),
        box_max_glsl[1], box_max_glsl[2], box_max_glsl[3], Int32(state.seed_radius),
        inv_dx2, inv_dy2, inv_dz2, state.current_s_boost,
        ep[1], ep[2], ep[3], ep[4], ep[5], ep[6],
        ep[7], ep[8], ep[9], ep[10], ep[11], ep[12]
    )
    
    # ── Diffusion steps (continue from current ping/pong state) ──
    cmd_bind_pipeline(cmd, PIPELINE_BIND_POINT_COMPUTE, state.pipeline_diffuse_step)
    
    is_ping = state.current_buffer_is_ping
    for k in 1:n_steps
        if is_ping
            cmd_bind_descriptor_sets(cmd, PIPELINE_BIND_POINT_COMPUTE, state.layout_diffuse_step, 0, [state.ds_diffuse_ping_to_pong], [])
        else
            cmd_bind_descriptor_sets(cmd, PIPELINE_BIND_POINT_COMPUTE, state.layout_diffuse_step, 0, [state.ds_diffuse_pong_to_ping], [])
        end
        ref_diffuse = Ref(pc_diffuse)
        GC.@preserve ref_diffuse begin
            cmd_push_constants(cmd, state.layout_diffuse_step, SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc_diffuse), Base.unsafe_convert(Ptr{Cvoid}, ref_diffuse))
        end
        cmd_dispatch(cmd, gx, gy, gz)
        cmd_pipeline_barrier(cmd, [barrier_compute], BufferMemoryBarrier[], ImageMemoryBarrier[];
                             src_stage_mask=PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             dst_stage_mask=PIPELINE_STAGE_COMPUTE_SHADER_BIT)
        is_ping = !is_ping
    end
    
    # ── Extract mask ──
    cmd_bind_pipeline(cmd, PIPELINE_BIND_POINT_COMPUTE, state.pipeline_extract_mask)
    ds_extract = is_ping ? state.ds_extract_from_ping : state.ds_extract_from_pong
    cmd_bind_descriptor_sets(cmd, PIPELINE_BIND_POINT_COMPUTE, state.layout_extract_mask, 0, [ds_extract], [])
    
    pc_extract = ExtractMaskPC(
        theta, tau, Int32(1), Int32(1),
        box_min_glsl[1], box_min_glsl[2], box_min_glsl[3], Int32(0),
        box_max_glsl[1], box_max_glsl[2], box_max_glsl[3], Int32(0)
    )
    ref_extract = Ref(pc_extract)
    GC.@preserve ref_extract begin
        cmd_push_constants(cmd, state.layout_extract_mask, SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc_extract), Base.unsafe_convert(Ptr{Cvoid}, ref_extract))
    end
    cmd_dispatch(cmd, gx, gy, gz)
    
    if copy_box_to_cpu && state.staging_box_buf !== nothing
        barrier_comp_to_xfer = MemoryBarrier(C_NULL, ACCESS_SHADER_WRITE_BIT, ACCESS_TRANSFER_READ_BIT)
        cmd_pipeline_barrier(cmd, [barrier_comp_to_xfer], BufferMemoryBarrier[], ImageMemoryBarrier[];
                             src_stage_mask=PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             dst_stage_mask=PIPELINE_STAGE_TRANSFER_BIT)
        
        region = BufferImageCopy(
            0, 0, 0,
            ImageSubresourceLayers(IMAGE_ASPECT_COLOR_BIT, 0, 0, 1),
            Offset3D(box_min_glsl[1], box_min_glsl[2], box_min_glsl[3]),
            Extent3D(bw, bh, bd)
        )
        cmd_copy_image_to_buffer(cmd, state.img_mask, IMAGE_LAYOUT_GENERAL, state.staging_box_buf, [region])
        
        barrier_xfer_to_host = MemoryBarrier(C_NULL, ACCESS_TRANSFER_WRITE_BIT, ACCESS_HOST_READ_BIT)
        cmd_pipeline_barrier(cmd, [barrier_xfer_to_host], BufferMemoryBarrier[], ImageMemoryBarrier[];
                             src_stage_mask=PIPELINE_STAGE_TRANSFER_BIT,
                             dst_stage_mask=PIPELINE_STAGE_HOST_BIT)
    else
        cmd_pipeline_barrier(cmd, [barrier_compute], BufferMemoryBarrier[], ImageMemoryBarrier[];
                             src_stage_mask=PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             dst_stage_mask=PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT)
    end
    
    unwrap(end_command_buffer(cmd))
    submit_info = SubmitInfo([], [], [cmd], [])
    unwrap(reset_fences(dev, [state.compute_fence]))
    unwrap(queue_submit(ctx.graphics_queue, [submit_info]; fence=state.compute_fence))
    unwrap(wait_for_fences(dev, [state.compute_fence], true, typemax(UInt64)))
    
    state.current_buffer_is_ping = is_ping
    state.accumulated_steps += n_steps
end

# ─── Cleanup ─────────────────────────────────────────────────────────────

function destroy_heat_diffusion!(state::HeatDiffusionState, ctx)
    if !state.is_initialized
        return
    end
    
    # Wait for GPU idle then let Vulkan.jl GC finalizers clean up resources
    # (Manual destroy_*/free_memory causes double-free on GC exit)
    unwrap(device_wait_idle(ctx.device))
    
    state.is_initialized = false
    state.diffusivity_uploaded = false
    if state.staging_box_mem !== nothing && state.staging_box_ptr != C_NULL
        try
            unmap_memory(ctx.device, state.staging_box_mem)
        catch
        end
        state.staging_box_ptr = C_NULL
    end
    @info "[HeatDiffusion] Resources marked for destruction."
end

# ─── one_time_submit! helper (uses the context's graphics queue) ─────────

function one_time_submit!(f::Function, ctx)
    alloc_info = CommandBufferAllocateInfo(ctx.command_pool, COMMAND_BUFFER_LEVEL_PRIMARY, 1)
    cmds = unwrap(Vulkan.allocate_command_buffers(ctx.device, alloc_info))
    cmd = cmds[1]
    begin_info = CommandBufferBeginInfo(flags = COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT)
    unwrap(begin_command_buffer(cmd, begin_info))
    f(cmd)
    unwrap(end_command_buffer(cmd))
    submit_info = SubmitInfo([], [], [cmd], [])
    unwrap(queue_submit(ctx.graphics_queue, [submit_info]))
    unwrap(queue_wait_idle(ctx.graphics_queue))
    free_command_buffers(ctx.device, ctx.command_pool, [cmd])
end


# ─── Read bounding box mask directly to seg_vol ──────────────────────────

"""
    apply_box_mask_to_segvol!(state, seg_vol::AbstractArray{T, 3}, label_val::T; theta=0.001044f0) where {T}

Applies the bounding-box mask retrieved from GPU directly into `seg_vol`.
Reads from the persistently mapped `staging_box_ptr` with ZERO allocations.
"""
function apply_box_mask_to_segvol!(state::HeatDiffusionState, seg_vol::AbstractArray{T, 3}, label_val::T; theta::Float32=0.001044f0) where {T}
    state.staging_box_ptr == C_NULL && return
    
    xmin, ymin, zmin = state.current_box_min
    xmax, ymax, zmax = state.current_box_max
    bw = xmax - xmin + 1
    bh = ymax - ymin + 1
    bd = zmax - zmin + 1
    
    ptr = Ptr{Float32}(state.staging_box_ptr)
    
    vox_count = 0
    @inbounds for kz in 1:bd
        z = zmin + kz - 1
        for ky in 1:bh
            y = ymin + ky - 1
            for kx in 1:bw
                x = xmin + kx - 1
                val = unsafe_load(ptr, kx + (ky - 1) * bw + (kz - 1) * bw * bh)
                if val > 0.5f0
                    seg_vol[x, y, z] = label_val
                    vox_count += 1
                end
            end
        end
    end
    return vox_count
end


# ─── GPU Direct Slice Rendering ──────────────────────────────────────────

"""
    render_slice_to_texture!(state, ctx, tex, slice_dim, slice_idx, active_id; theta=0.001f0)

Directly computes a 2D slice from the 3D heat mask and writes it to `tex.image` (a VkTexture).
Eliminates CPU roundtrips completely for UI preview rendering.
"""
function render_slice_to_texture!(state::HeatDiffusionState, ctx, tex::Any, slice_dim::Int, slice_idx::Int, active_id::UInt32; theta::Float32=0.001f0)
    dev = ctx.device
    is_ping = state.current_buffer_is_ping
    ds = is_ping ? state.ds_slice_ping : state.ds_slice_pong
    
    # 1. Update descriptor set with the target texture view
    di = DescriptorImageInfo(state.dummy_sampler, tex.view, IMAGE_LAYOUT_GENERAL)
    write = WriteDescriptorSet(ds, 1, 0, DESCRIPTOR_TYPE_STORAGE_IMAGE, [di], [], [])
    update_descriptor_sets(dev, [write], [])
    
    # 2. Dispatch
    cmd = state.compute_cmd_buffer
    unwrap(reset_command_buffer(cmd))
    begin_info = CommandBufferBeginInfo(flags = COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT)
    unwrap(begin_command_buffer(cmd, begin_info))
    
    barrier_to_general = ImageMemoryBarrier(
        C_NULL, ACCESS_SHADER_READ_BIT, ACCESS_SHADER_WRITE_BIT,
        IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, IMAGE_LAYOUT_GENERAL,
        0, 0, tex.image, ImageSubresourceRange(IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1)
    )
    cmd_pipeline_barrier(cmd, MemoryBarrier[], BufferMemoryBarrier[], [barrier_to_general];
                         src_stage_mask=PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         dst_stage_mask=PIPELINE_STAGE_COMPUTE_SHADER_BIT)
    
    cmd_bind_pipeline(cmd, PIPELINE_BIND_POINT_COMPUTE, state.pipeline_slice)
    cmd_bind_descriptor_sets(cmd, PIPELINE_BIND_POINT_COMPUTE, state.layout_slice, 0, [ds], [])
    
    pc_u32 = UInt32[reinterpret(UInt32, theta), UInt32(slice_dim), UInt32(slice_idx), active_id]
    GC.@preserve pc_u32 begin
        cmd_push_constants(cmd, state.layout_slice, SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc_u32), Ptr{Cvoid}(pointer(pc_u32)))
    end
    
    gx = cld(tex.width, 8)
    gy = cld(tex.height, 8)
    cmd_dispatch(cmd, gx, gy, 1)
    
    barrier_to_read = ImageMemoryBarrier(
        C_NULL, ACCESS_SHADER_WRITE_BIT, ACCESS_SHADER_READ_BIT,
        IMAGE_LAYOUT_GENERAL, IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        0, 0, tex.image, ImageSubresourceRange(IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1)
    )
    cmd_pipeline_barrier(cmd, MemoryBarrier[], BufferMemoryBarrier[], [barrier_to_read];
                         src_stage_mask=PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         dst_stage_mask=PIPELINE_STAGE_FRAGMENT_SHADER_BIT)
                         
    unwrap(end_command_buffer(cmd))
    submit_info = SubmitInfo([], [], [cmd], [])
    unwrap(reset_fences(dev, [state.compute_fence]))
    unwrap(queue_submit(ctx.graphics_queue, [submit_info]; fence=state.compute_fence))
    unwrap(wait_for_fences(dev, [state.compute_fence], true, typemax(UInt64)))
end

end # module VulkanHeatDiffusion
