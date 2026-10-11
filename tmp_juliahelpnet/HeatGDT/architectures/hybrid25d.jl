# architectures/hybrid25d.jl — Hybrid 2.5D/3D Multi-Planar Fusion Network
# Processes 3 orthogonal planes with plane-specific kernels, then fuses in 3D

using Lux
using NNlib

"""
    PlanarBranch(kernel, in_c, channels)

Single planar processing branch using plane-specific asymmetric convolution kernels.
E.g., `(3,3,1)` for XY plane, `(3,1,3)` for XZ plane, `(1,3,3)` for YZ plane.
"""
struct PlanarBranch <: Lux.AbstractLuxContainerLayer{(:l1, :l2, :l3)}
    l1; l2; l3
end

function PlanarBranch(kernel::NTuple{3,Int}, in_c::Int, pad_spec::NTuple{3,Int})
    l1 = Chain(
        Conv(kernel, in_c => 16, pad=pad_spec),
        BatchNorm(16, momentum=0.01f0, epsilon=1f-3),
        x -> relu.(x)
    )
    l2 = Chain(
        Conv(kernel, 16 => 32, pad=pad_spec),
        BatchNorm(32, momentum=0.01f0, epsilon=1f-3),
        x -> relu.(x)
    )
    l3 = Chain(
        Conv(kernel, 32 => 16, pad=pad_spec),
        BatchNorm(16, momentum=0.01f0, epsilon=1f-3),
        x -> relu.(x)
    )
    return PlanarBranch(l1, l2, l3)
end

function (m::PlanarBranch)(x, ps, st)
    h1, st_l1 = m.l1(x, ps.l1, st.l1)
    h2, st_l2 = m.l2(h1, ps.l2, st.l2)
    h3, st_l3 = m.l3(h2, ps.l3, st.l3)
    st_new = (l1=st_l1, l2=st_l2, l3=st_l3)
    return h3, st_new
end

"""
    Hybrid25D3D(in_ch=3)

Hybrid 2.5D/3D Multi-Planar Network:
- XY branch: Conv(3,3,1) kernels — axial plane processing
- XZ branch: Conv(3,1,3) kernels — coronal plane processing
- YZ branch: Conv(1,3,3) kernels — sagittal plane processing
Each branch produces 16 channels → concatenate (48ch) → 3D Conv fusion → 1ch output.

Output: `(X, Y, Z, 1, B)` edge probability map.
"""
struct Hybrid25D3D <: Lux.AbstractLuxContainerLayer{(:xy_branch, :xz_branch, :yz_branch, :fuse)}
    xy_branch; xz_branch; yz_branch
    fuse
end

function Hybrid25D3D(in_ch::Int=3)
    xy_branch = PlanarBranch((3, 3, 1), in_ch, (1, 1, 0))
    xz_branch = PlanarBranch((3, 1, 3), in_ch, (1, 0, 1))
    yz_branch = PlanarBranch((1, 3, 3), in_ch, (0, 1, 1))

    fuse = Chain(
        Conv((3, 3, 3), 48 => 32, pad=1),
        BatchNorm(32, momentum=0.01f0, epsilon=1f-3),
        x -> relu.(x),
        Conv((3, 3, 3), 32 => 16, pad=1),
        BatchNorm(16, momentum=0.01f0, epsilon=1f-3),
        x -> relu.(x),
        Conv((1, 1, 1), 16 => 1),
        x -> sigmoid.(x)
    )
    return Hybrid25D3D(xy_branch, xz_branch, yz_branch, fuse)
end

function (m::Hybrid25D3D)(x, ps, st)
    # Process each orthogonal plane
    xy_out, st_xy = m.xy_branch(x, ps.xy_branch, st.xy_branch)
    xz_out, st_xz = m.xz_branch(x, ps.xz_branch, st.xz_branch)
    yz_out, st_yz = m.yz_branch(x, ps.yz_branch, st.yz_branch)

    # Fuse: concatenate 3×16=48 channels → 3D conv → sigmoid
    combined = cat(xy_out, xz_out, yz_out, dims=4)
    edge_out, st_fuse = m.fuse(combined, ps.fuse, st.fuse)

    st_new = (
        xy_branch=st_xy, xz_branch=st_xz, yz_branch=st_yz,
        fuse=st_fuse
    )
    return edge_out, st_new
end
