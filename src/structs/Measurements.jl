module Measurements
using Base: @kwdef

export SphereMeasurement, LineMeasurement, MeasurementData, compute_measurement_vertices, compute_line_vertices
export MEASUREMENT_COLORS, get_measurement_color

# ─── Color Palette ────────────────────────────────────────────────────────────
const MEASUREMENT_COLORS = [
    (1.0f0, 0.3f0, 0.3f0, 0.8f0),  # 1 Red
    (0.3f0, 0.85f0, 0.3f0, 0.8f0), # 2 Green
    (0.4f0, 0.55f0, 1.0f0, 0.8f0), # 3 Blue
    (1.0f0, 0.85f0, 0.0f0, 0.8f0), # 4 Yellow
    (1.0f0, 0.5f0, 0.0f0, 0.8f0),  # 5 Orange
    (0.85f0, 0.3f0, 0.85f0, 0.8f0),# 6 Purple
    (0.0f0, 0.85f0, 0.85f0, 0.8f0),# 7 Cyan
    (1.0f0, 0.6f0, 0.7f0, 0.8f0),  # 8 Pink
]

function get_measurement_color(idx::Int)
    c = MEASUREMENT_COLORS[((idx - 1) % length(MEASUREMENT_COLORS)) + 1]
    return Float32[c[1], c[2], c[3], c[4]]
end

# ─── Structs ──────────────────────────────────────────────────────────────────
@kwdef mutable struct SphereMeasurement
    id::Int
    center_idx::NTuple{3, Float32} # (x, y, z) in primary axial voxel coordinates
    radius_mm::Float32
    radius_x_mm::Float32 = 0.0f0  # 0 = use radius_mm (per-axis for ellipsoid)
    radius_y_mm::Float32 = 0.0f0
    radius_z_mm::Float32 = 0.0f0
    suv_mean::Float32
    suv_max::Float32
    is_active::Bool = false # true if currently tracking mouse
    color_idx::Int = 1
    editing_axis::Symbol = :none  # :none, :x, :y, :z when edge-dragging
end

@kwdef mutable struct LineMeasurement
    id::Int
    start_idx::NTuple{3, Float32}  # (x, y, z) in primary axial voxel coordinates
    end_idx::NTuple{3, Float32}    # (x, y, z) endpoint
    length_mm::Float32 = 0.0f0     # Euclidean distance in mm
    suv_mean::Float32 = 0.0f0
    suv_max::Float32 = 0.0f0
    is_active::Bool = false        # true while placing (only start set, end follows cursor)
    color_idx::Int = 1
    editing_endpoint::Symbol = :none  # :none, :start, :end when editing
end

const MeasurementData = Union{SphereMeasurement, LineMeasurement}

# ─── Serialization for HDF5 persistence ──────────────────────────────────────
export serialize_measurements, deserialize_measurements

"""Serialize spheres + lines into a string for HDF5 attribute storage."""
function serialize_measurements(spheres::Vector{SphereMeasurement}, lines::Vector{LineMeasurement})::String
    parts = String[]
    for s in spheres
        (s.is_active || s.id < 1) && continue  # Don't save active or ghost (id=-1) measurements
        push!(parts, "S|$(s.id)|$(s.center_idx[1]),$(s.center_idx[2]),$(s.center_idx[3])|$(s.radius_mm)|$(s.radius_x_mm),$(s.radius_y_mm),$(s.radius_z_mm)|$(s.suv_mean)|$(s.suv_max)|$(s.color_idx)")
    end
    for l in lines
        (l.is_active || l.id < 1) && continue
        push!(parts, "L|$(l.id)|$(l.start_idx[1]),$(l.start_idx[2]),$(l.start_idx[3])|$(l.end_idx[1]),$(l.end_idx[2]),$(l.end_idx[3])|$(l.length_mm)|$(l.suv_mean)|$(l.suv_max)|$(l.color_idx)")
    end
    return join(parts, "\n")
end

"""Deserialize from string into (spheres, lines). Returns empty vectors on error."""
function deserialize_measurements(s::String)::Tuple{Vector{SphereMeasurement}, Vector{LineMeasurement}}
    spheres = SphereMeasurement[]
    lines = LineMeasurement[]
    isempty(s) && return (spheres, lines)
    for line in split(s, "\n")
        isempty(line) && continue
        fields = split(line, "|")
        try
            if fields[1] == "S" && length(fields) >= 8
                cparts = split(fields[3], ",")
                center = (parse(Float32, cparts[1]), parse(Float32, cparts[2]), parse(Float32, cparts[3]))
                radius_mm = parse(Float32, fields[4])
                rparts = split(fields[5], ",")
                rx = parse(Float32, rparts[1])
                ry = parse(Float32, rparts[2])
                rz = parse(Float32, rparts[3])
                push!(spheres, SphereMeasurement(
                    id = parse(Int, fields[2]),
                    center_idx = center,
                    radius_mm = radius_mm,
                    radius_x_mm = rx, radius_y_mm = ry, radius_z_mm = rz,
                    suv_mean = parse(Float32, fields[6]),
                    suv_max = parse(Float32, fields[7]),
                    is_active = false,
                    color_idx = parse(Int, fields[8])
                ))
            elseif fields[1] == "L" && length(fields) >= 8
                sparts = split(fields[3], ",")
                start_pt = (parse(Float32, sparts[1]), parse(Float32, sparts[2]), parse(Float32, sparts[3]))
                eparts = split(fields[4], ",")
                end_pt = (parse(Float32, eparts[1]), parse(Float32, eparts[2]), parse(Float32, eparts[3]))
                push!(lines, LineMeasurement(
                    id = parse(Int, fields[2]),
                    start_idx = start_pt,
                    end_idx = end_pt,
                    length_mm = parse(Float32, fields[5]),
                    suv_mean = parse(Float32, fields[6]),
                    suv_max = parse(Float32, fields[7]),
                    is_active = false,
                    color_idx = parse(Int, fields[8])
                ))
            end
        catch e
            @warn "Failed to parse measurement: $line" exception=e
        end
    end
    return (spheres, lines)
end

# ─── Helper: thick line as 2-triangle quad ────────────────────────────────────
function _push_thick_line!(vertices::Vector{Float32}, u1, v1, u2, v2, color, w, h, thickness::Float32=0.005f0)
    dx = u2 - u1
    dy = v2 - v1
    len = sqrt(dx^2 + dy^2)
    if len > 1e-6
        nx = -dy / len
        ny = dx / len
        
        t_u = thickness
        t_v = thickness * (w > 0 ? (w / h) : 1.0f0)
        
        ox = nx * t_u
        oy = ny * t_v
        
        # 6 vertices for a 2-triangle quad
        push!(vertices, u1 + ox, v1 + oy, color[1], color[2], color[3], color[4])
        push!(vertices, u2 + ox, v2 + oy, color[1], color[2], color[3], color[4])
        push!(vertices, u1 - ox, v1 - oy, color[1], color[2], color[3], color[4])
        
        push!(vertices, u1 - ox, v1 - oy, color[1], color[2], color[3], color[4])
        push!(vertices, u2 + ox, v2 + oy, color[1], color[2], color[3], color[4])
        push!(vertices, u2 - ox, v2 - oy, color[1], color[2], color[3], color[4])
    end
end

# ─── Helper: small crosshair at a point ───────────────────────────────────────
function _push_crosshair!(vertices::Vector{Float32}, u, v, color, w, h, size::Float32=0.015f0, thickness::Float32=0.003f0)
    _push_thick_line!(vertices, u - size, v, u + size, v, color, w, h, thickness)
    _push_thick_line!(vertices, u, v - size * (w > 0 ? (w/h) : 1.0f0), u, v + size * (w > 0 ? (w/h) : 1.0f0), color, w, h, thickness)
end

# ─── Helper: small endpoint handle (small filled square) ─────────────────────
function _push_endpoint_handle!(vertices::Vector{Float32}, u, v, color, w, h, size::Float32=0.008f0)
    # Draw a small + symbol at the endpoint to indicate it's interactive
    _push_thick_line!(vertices, u - size, v, u + size, v, color, w, h, 0.004f0)
    _push_thick_line!(vertices, u, v - size * (w > 0 ? (w/h) : 1.0f0), u, v + size * (w > 0 ? (w/h) : 1.0f0), color, w, h, 0.004f0)
end

# ─── Helper: get effective per-axis radii ─────────────────────────────────────
function _get_effective_radii(m::SphereMeasurement)
    rx = m.radius_x_mm > 0 ? m.radius_x_mm : m.radius_mm
    ry = m.radius_y_mm > 0 ? m.radius_y_mm : m.radius_mm
    rz = m.radius_z_mm > 0 ? m.radius_z_mm : m.radius_mm
    return (rx, ry, rz)
end

"""
    compute_measurement_vertices(measurements, state, panel_id)

Generates the vector buffer (vec2 pos, vec4 color) for all sphere measurements
intersecting the current slice shown on `panel_id`.
No text labels — each measurement uses its assigned color.
"""
function compute_measurement_vertices(measurements::Vector{SphereMeasurement}, state, panel_id::Int)::Vector{Float32}
    vertices = Float32[]
    panelState = state[panel_id]
    currentSlice = panelState.currentDisplayedSlice
    w = Float32(panelState.calcDimsStruct.imageTextureWidth)
    h = Float32(panelState.calcDimsStruct.imageTextureHeight)
    
    if w <= 0 || h <= 0
        return vertices
    end
    
    # Use state[1] (Axial view) which is guaranteed to hold the original unpermuted spacings
    orig_sp = state[1].spacingsValue[1]
    spacing_x, spacing_y, spacing_z = Float32(orig_sp[1]), Float32(orig_sp[2]), Float32(orig_sp[3])
    if spacing_x <= 0; spacing_x = 1.0f0; end
    if spacing_y <= 0; spacing_y = 1.0f0; end
    if spacing_z <= 0; spacing_z = 1.0f0; end
    
    for m in measurements
        cx, cy, cz = m.center_idx
        rx, ry, rz = _get_effective_radii(m)
        
        # Determine distance 'd' from center to slice plane based on panel orientation
        d_mm = 0.0f0
        R_axis = 0.0f0  # radius along the slice-normal axis
        if panel_id == 1 || panel_id == 2 || panel_id == 5
            d_mm = abs(cz - currentSlice) * spacing_z
            R_axis = rz
        elseif panel_id == 3
            d_mm = abs(cx - currentSlice) * spacing_x
            R_axis = rx
        elseif panel_id == 4
            d_mm = abs(cy - currentSlice) * spacing_y
            R_axis = ry
        end
        
        if d_mm < R_axis
            # Color from palette (no text labels)
            color = get_measurement_color(m.color_idx)
            # Make active measurements slightly brighter / more opaque
            if m.is_active
                color[4] = 1.0f0
            end
            
            # Ellipse intersection: parametric approach
            # Fraction along the normal axis
            frac = d_mm / R_axis
            scale_factor = sqrt(1.0f0 - frac^2)
            
            # Determine the 2D radii in the slice plane (in mm)
            r_horiz_mm, r_vert_mm = 0.0f0, 0.0f0
            if panel_id == 1 || panel_id == 2 || panel_id == 5
                r_horiz_mm = rx * scale_factor
                r_vert_mm = ry * scale_factor
            elseif panel_id == 3
                r_horiz_mm = ry * scale_factor
                r_vert_mm = rz * scale_factor
            elseif panel_id == 4
                r_horiz_mm = rx * scale_factor
                r_vert_mm = rz * scale_factor
            end
            
            n_segments = 64
            for i in 1:n_segments
                t1 = (i - 1) / n_segments * 2pi
                t2 = i / n_segments * 2pi
                
                px_vox_1, py_vox_1 = 0.0f0, 0.0f0
                px_vox_2, py_vox_2 = 0.0f0, 0.0f0
                
                for (j, t) in enumerate((t1, t2))
                    off_x_mm = r_horiz_mm * cos(t)
                    off_y_mm = r_vert_mm * sin(t)
                    
                    if panel_id == 1 || panel_id == 2 || panel_id == 5
                        px = cx + off_x_mm / spacing_x
                        py = cy + off_y_mm / spacing_y
                    elseif panel_id == 3
                        px = cy + off_x_mm / spacing_y
                        py = cz + off_y_mm / spacing_z
                    elseif panel_id == 4
                        px = cx + off_x_mm / spacing_x
                        py = cz + off_y_mm / spacing_z
                    else
                        px = 0.0f0; py = 0.0f0
                    end
                    
                    if j == 1
                        px_vox_1, py_vox_1 = px, py
                    else
                        px_vox_2, py_vox_2 = px, py
                    end
                end
                
                u1 = (px_vox_1 - 0.5f0) / w
                v1 = (py_vox_1 - 0.5f0) / h
                u2 = (px_vox_2 - 0.5f0) / w
                v2 = (py_vox_2 - 0.5f0) / h
                
                _push_thick_line!(vertices, u1, v1, u2, v2, color, w, h, 0.002f0)
            end
        end
    end
    
    return vertices
end

"""
    _project_point_to_uv(px, py, pz, panel_id, w, h) -> (u, v)

Project a 3D voxel coordinate to panel UV space.
"""
function _project_point_to_uv(px::Float32, py::Float32, pz::Float32,
                               panel_id::Int, w::Float32, h::Float32)
    img_x, img_y = 0.0f0, 0.0f0
    if panel_id == 1 || panel_id == 2 || panel_id == 5
        img_x, img_y = px, py
    elseif panel_id == 3
        img_x, img_y = py, pz
    elseif panel_id == 4
        img_x, img_y = px, pz
    end
    u = (img_x - 0.5f0) / w
    v = (img_y - 0.5f0) / h
    return (u, v)
end

"""
    compute_line_vertices(line_measurements, state, panel_id)

Generates vector buffer for all line measurements visible on the current slice.
Includes endpoint handles for editing, crosshair for zero-length lines.
No text labels — each measurement uses its assigned color.
"""
function compute_line_vertices(line_measurements::Vector{LineMeasurement}, state, panel_id::Int)::Vector{Float32}
    vertices = Float32[]
    
    panelState = state[panel_id]
    currentSlice = panelState.currentDisplayedSlice
    w = Float32(panelState.calcDimsStruct.imageTextureWidth)
    h = Float32(panelState.calcDimsStruct.imageTextureHeight)
    
    if w <= 0 || h <= 0
        return vertices
    end
    
    tolerance = 2.0f0  # show lines within ±2 voxels of current slice
    
    for lm in line_measurements
        sx, sy, sz = lm.start_idx
        ex, ey, ez = lm.end_idx
        
        # Check if either endpoint is near the current slice
        visible = false
        if panel_id == 1 || panel_id == 2 || panel_id == 5
            visible = abs(sz - currentSlice) <= tolerance || abs(ez - currentSlice) <= tolerance
        elseif panel_id == 3
            visible = abs(sx - currentSlice) <= tolerance || abs(ex - currentSlice) <= tolerance
        elseif panel_id == 4
            visible = abs(sy - currentSlice) <= tolerance || abs(ey - currentSlice) <= tolerance
        end
        
        if visible
            color = get_measurement_color(lm.color_idx)
            if lm.is_active
                color[4] = 1.0f0
            end
            
            u1, v1 = _project_point_to_uv(sx, sy, sz, panel_id, w, h)
            u2, v2 = _project_point_to_uv(ex, ey, ez, panel_id, w, h)
            
            if sx == ex && sy == ey && sz == ez
                # Zero-length line — draw crosshair at start point to show placement
                _push_crosshair!(vertices, u1, v1, color, w, h)
            else
                # Normal line
                _push_thick_line!(vertices, u1, v1, u2, v2, color, w, h, 0.004f0)
                
                # Draw endpoint handles (small crosses at each end)
                _push_endpoint_handle!(vertices, u1, v1, color, w, h)
                _push_endpoint_handle!(vertices, u2, v2, color, w, h)
            end
        end
    end
    
    return vertices
end

end
