# architectures/dframenet.jl — Direction-Field Network (D-FRAME Style)
# Shared encoder with dual-head output: edges + 3D direction/advection field

using Lux
using NNlib

"""
    DFrameNet(in_ch=3)

Direction-Field Network with shared 3-stage encoder producing two heads:
- **Edge Head**: 1-channel edge probability → sigmoid → `(X, Y, Z, 1, B)`
- **Direction Head**: 3-channel direction field → `(X, Y, Z, 3, B)` (unnormalized)

The direction field v(x) represents the advection velocity for the
reaction-diffusion-advection PDE.

Output: `NamedTuple{(:edges, :direction)}` with shapes `(X,Y,Z,1,B)` and `(X,Y,Z,3,B)`.
"""
struct DFrameNet <: Lux.AbstractLuxContainerLayer{(:e1, :e2, :e3, :d2, :d1, :up2, :up1, :edge_head, :dir_head)}
    e1; e2; e3
    d2; d1
    up2; up1
    edge_head
    dir_head
end

function _dframe_conv_block(in_c::Int, out_c::Int)
    Chain(
        Conv((3, 3, 3), in_c => out_c, pad=1),
        BatchNorm(out_c, momentum=0.01f0, epsilon=1f-3),
        x -> relu.(x),
        Conv((3, 3, 3), out_c => out_c, pad=1),
        BatchNorm(out_c, momentum=0.01f0, epsilon=1f-3),
        x -> relu.(x)
    )
end

function DFrameNet(in_ch::Int=3)
    # Shared encoder
    e1 = _dframe_conv_block(in_ch, 32)
    e2 = _dframe_conv_block(32, 64)
    e3 = _dframe_conv_block(64, 128)

    # Shared decoder
    d2 = _dframe_conv_block(128 + 64, 64)
    d1 = _dframe_conv_block(64 + 32, 32)
    up2 = Upsample(:trilinear, scale=2)
    up1 = Upsample(:trilinear, scale=2)

    # Edge head: 32→1 sigmoid
    edge_head = Chain(
        Conv((3, 3, 3), 32 => 16, pad=1),
        BatchNorm(16, momentum=0.01f0, epsilon=1f-3),
        x -> relu.(x),
        Conv((1, 1, 1), 16 => 1),
        x -> sigmoid.(x)
    )

    # Direction head: 32→3 (raw direction field, no activation)
    dir_head = Chain(
        Conv((3, 3, 3), 32 => 16, pad=1),
        BatchNorm(16, momentum=0.01f0, epsilon=1f-3),
        x -> relu.(x),
        Conv((1, 1, 1), 16 => 3)
    )

    return DFrameNet(e1, e2, e3, d2, d1, up2, up1, edge_head, dir_head)
end

function (m::DFrameNet)(x, ps, st)
    # Encoder
    x1, st_e1 = m.e1(x, ps.e1, st.e1)
    p1 = NNlib.maxpool(x1, (2, 2, 2))
    x2, st_e2 = m.e2(p1, ps.e2, st.e2)
    p2 = NNlib.maxpool(x2, (2, 2, 2))
    x3, st_e3 = m.e3(p2, ps.e3, st.e3)

    # Decoder
    u2, st_up2 = m.up2(x3, ps.up2, st.up2)
    dec2, st_d2 = m.d2(cat(u2, x2, dims=4), ps.d2, st.d2)
    u1, st_up1 = m.up1(dec2, ps.up1, st.up1)
    dec1, st_d1 = m.d1(cat(u1, x1, dims=4), ps.d1, st.d1)

    # Dual heads
    edges, st_eh = m.edge_head(dec1, ps.edge_head, st.edge_head)
    direction, st_dh = m.dir_head(dec1, ps.dir_head, st.dir_head)

    st_new = (
        e1=st_e1, e2=st_e2, e3=st_e3,
        d2=st_d2, d1=st_d1,
        up2=st_up2, up1=st_up1,
        edge_head=st_eh, dir_head=st_dh
    )
    return (edges=edges, direction=direction), st_new
end
