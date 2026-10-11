# architectures/dexined3d.jl — DexiNed-3D: Dense Extreme Inception for 3D Edge Detection
# Multi-branch inception blocks with dense connections and side outputs

using Lux
using NNlib

"""
    InceptionBlock3D(in_c, out_c)

Single extreme inception block with 3 parallel branches:
- Branch A: Conv(1,1,1) — point-wise
- Branch B: Conv(3,3,3, pad=1) — local spatial
- Branch C: Conv(3,3,3, pad=2, dilation=2) — dilated context
Concatenates branches, then fuses with 1×1×1 conv.
"""
struct InceptionBlock3D <: Lux.AbstractLuxContainerLayer{(:branch_a, :branch_b, :branch_c, :fuse)}
    branch_a; branch_b; branch_c; fuse
end

function InceptionBlock3D(in_c::Int, out_c::Int)
    bc = div(out_c, 3)  # channels per branch
    bc_rem = out_c - 2 * bc  # give remainder to branch_a

    branch_a = Chain(
        Conv((1, 1, 1), in_c => bc_rem),
        BatchNorm(bc_rem, momentum=0.01f0, epsilon=1f-3),
        x -> relu.(x)
    )
    branch_b = Chain(
        Conv((3, 3, 3), in_c => bc, pad=1),
        BatchNorm(bc, momentum=0.01f0, epsilon=1f-3),
        x -> relu.(x)
    )
    branch_c = Chain(
        Conv((3, 3, 3), in_c => bc, pad=2, dilation=2),
        BatchNorm(bc, momentum=0.01f0, epsilon=1f-3),
        x -> relu.(x)
    )
    fuse = Chain(
        Conv((1, 1, 1), out_c => out_c),
        BatchNorm(out_c, momentum=0.01f0, epsilon=1f-3),
        x -> relu.(x)
    )
    return InceptionBlock3D(branch_a, branch_b, branch_c, fuse)
end

function (m::InceptionBlock3D)(x, ps, st)
    a, st_a = m.branch_a(x, ps.branch_a, st.branch_a)
    b, st_b = m.branch_b(x, ps.branch_b, st.branch_b)
    c, st_c = m.branch_c(x, ps.branch_c, st.branch_c)
    combined = cat(a, b, c, dims=4)
    out, st_f = m.fuse(combined, ps.fuse, st.fuse)
    st_new = (branch_a=st_a, branch_b=st_b, branch_c=st_c, fuse=st_f)
    return out, st_new
end

"""
    DexiNed3D(in_ch=3)

DexiNed-3D with 4 inception blocks [16, 32, 48, 64] with dense connections.
Each block receives concatenated input from all previous blocks.
Side outputs at each scale, with final fusion head.

Output: `(X, Y, Z, 1, B)` edge probability map.
"""
struct DexiNed3D <: Lux.AbstractLuxContainerLayer{(:stem, :b1, :b2, :b3, :b4, :s1, :s2, :s3, :s4, :fuse)}
    stem; b1; b2; b3; b4
    s1; s2; s3; s4
    fuse
end

function DexiNed3D(in_ch::Int=3)
    stem_ch = 16
    stem = Chain(
        Conv((3, 3, 3), in_ch => stem_ch, pad=1),
        BatchNorm(stem_ch, momentum=0.01f0, epsilon=1f-3),
        x -> relu.(x)
    )

    # Dense inception blocks — each receives concat of all previous outputs
    b1 = InceptionBlock3D(stem_ch, 16)           # input: 16 → output: 16
    b2 = InceptionBlock3D(stem_ch + 16, 32)      # input: 16+16=32 → output: 32
    b3 = InceptionBlock3D(stem_ch + 16 + 32, 48) # input: 64 → output: 48
    b4 = InceptionBlock3D(stem_ch + 16 + 32 + 48, 64) # input: 112 → output: 64

    # Side outputs from each block
    s1 = Conv((1, 1, 1), 16 => 1)
    s2 = Conv((1, 1, 1), 32 => 1)
    s3 = Conv((1, 1, 1), 48 => 1)
    s4 = Conv((1, 1, 1), 64 => 1)

    # Final fusion head
    fuse = Chain(
        Conv((3, 3, 3), 4 => 8, pad=1),
        BatchNorm(8, momentum=0.01f0, epsilon=1f-3),
        x -> relu.(x),
        Conv((1, 1, 1), 8 => 1),
        x -> sigmoid.(x)
    )
    return DexiNed3D(stem, b1, b2, b3, b4, s1, s2, s3, s4, fuse)
end

function (m::DexiNed3D)(x, ps, st)
    # Stem
    h0, st_stem = m.stem(x, ps.stem, st.stem)

    # Block 1
    o1, st_b1 = m.b1(h0, ps.b1, st.b1)
    side1, st_s1 = m.s1(o1, ps.s1, st.s1)

    # Block 2 (dense: receives h0 ⊕ o1)
    o2, st_b2 = m.b2(cat(h0, o1, dims=4), ps.b2, st.b2)
    side2, st_s2 = m.s2(o2, ps.s2, st.s2)

    # Block 3 (dense: receives h0 ⊕ o1 ⊕ o2)
    o3, st_b3 = m.b3(cat(h0, o1, o2, dims=4), ps.b3, st.b3)
    side3, st_s3 = m.s3(o3, ps.s3, st.s3)

    # Block 4 (dense: receives h0 ⊕ o1 ⊕ o2 ⊕ o3)
    o4, st_b4 = m.b4(cat(h0, o1, o2, o3, dims=4), ps.b4, st.b4)
    side4, st_s4 = m.s4(o4, ps.s4, st.s4)

    # Fuse side outputs
    cat_sides = cat(side1, side2, side3, side4, dims=4)
    edge_out, st_fuse = m.fuse(cat_sides, ps.fuse, st.fuse)

    st_new = (
        stem=st_stem,
        b1=st_b1, b2=st_b2, b3=st_b3, b4=st_b4,
        s1=st_s1, s2=st_s2, s3=st_s3, s4=st_s4,
        fuse=st_fuse
    )
    return edge_out, st_new
end
