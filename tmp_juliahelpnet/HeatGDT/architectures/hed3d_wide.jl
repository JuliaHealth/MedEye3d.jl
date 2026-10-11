# architectures/hed3d_wide.jl — Wider 3D Holistically-Nested Edge Detector
# 5-stage VGG-style encoder with channels [48, 96, 192, 384, 768]
# ~31.8M parameters (vs 14.1M for standard HED3D [32, 64, 128, 256, 512])
# Same topology as HED3D, just 1.5x wider at each stage.

using Lux
using NNlib

"""
    HED3D_Wide(in_ch=3)

Wider 3D Holistically-Nested Edge Detector with 5 encoder stages
[in→48→96→192→384→768], each with a 1×1×1 side output upsampled
to input resolution, and a final weighted fusion head.

~31.8M parameters. Output: `(X, Y, Z, 1, B)` edge probability map.
"""
struct HED3D_Wide <: Lux.AbstractLuxContainerLayer{(:e1, :e2, :e3, :e4, :e5, :s1, :s2, :s3, :s4, :s5, :fuse)}
    e1; e2; e3; e4; e5
    s1; s2; s3; s4; s5
    fuse
end

function _hed_wide_conv_block(in_c::Int, out_c::Int)
    Chain(
        Conv((3, 3, 3), in_c => out_c, pad=1),
        BatchNorm(out_c, momentum=0.01f0, epsilon=1f-3),
        x -> relu.(x),
        Conv((3, 3, 3), out_c => out_c, pad=1),
        BatchNorm(out_c, momentum=0.01f0, epsilon=1f-3),
        x -> relu.(x)
    )
end

function HED3D_Wide(in_ch::Int=3)
    # 1.5x wider channels: 48, 96, 192, 384, 768
    e1 = _hed_wide_conv_block(in_ch, 48)
    e2 = _hed_wide_conv_block(48, 96)
    e3 = _hed_wide_conv_block(96, 192)
    e4 = _hed_wide_conv_block(192, 384)
    e5 = _hed_wide_conv_block(384, 768)

    # Side output heads: 1×1×1 conv producing 1-channel map
    s1 = Conv((1, 1, 1), 48 => 1)
    s2 = Chain(Conv((1, 1, 1), 96 => 1), Upsample(:trilinear, scale=2))
    s3 = Chain(Conv((1, 1, 1), 192 => 1), Upsample(:trilinear, scale=4))
    s4 = Chain(Conv((1, 1, 1), 384 => 1), Upsample(:trilinear, scale=8))
    s5 = Chain(Conv((1, 1, 1), 768 => 1), Upsample(:trilinear, scale=16))

    # Weighted fusion: concat 5 side outputs (5ch) → 1×1×1 → sigmoid
    fuse = Chain(
        Conv((1, 1, 1), 5 => 1),
        x -> sigmoid.(x)
    )
    return HED3D_Wide(e1, e2, e3, e4, e5, s1, s2, s3, s4, s5, fuse)
end

function (m::HED3D_Wide)(x, ps, st)
    # x: (X, Y, Z, C, B)

    # Stage 1 — full resolution
    x1, st_e1 = m.e1(x, ps.e1, st.e1)
    side1, st_s1 = m.s1(x1, ps.s1, st.s1)

    # Stage 2 — 1/2 resolution
    p1 = NNlib.maxpool(x1, (2, 2, 2))
    x2, st_e2 = m.e2(p1, ps.e2, st.e2)
    side2, st_s2 = m.s2(x2, ps.s2, st.s2)

    # Stage 3 — 1/4 resolution
    p2 = NNlib.maxpool(x2, (2, 2, 2))
    x3, st_e3 = m.e3(p2, ps.e3, st.e3)
    side3, st_s3 = m.s3(x3, ps.s3, st.s3)

    # Stage 4 — 1/8 resolution
    p3 = NNlib.maxpool(x3, (2, 2, 2))
    x4, st_e4 = m.e4(p3, ps.e4, st.e4)
    side4, st_s4 = m.s4(x4, ps.s4, st.s4)

    # Stage 5 — 1/16 resolution
    p4 = NNlib.maxpool(x4, (2, 2, 2))
    x5, st_e5 = m.e5(p4, ps.e5, st.e5)
    side5, st_s5 = m.s5(x5, ps.s5, st.s5)

    # Fuse all side outputs
    cat_sides = cat(side1, side2, side3, side4, side5, dims=4)
    edge_out, st_fuse = m.fuse(cat_sides, ps.fuse, st.fuse)

    st_new = (
        e1=st_e1, e2=st_e2, e3=st_e3, e4=st_e4, e5=st_e5,
        s1=st_s1, s2=st_s2, s3=st_s3, s4=st_s4, s5=st_s5,
        fuse=st_fuse
    )
    return edge_out, st_new
end
