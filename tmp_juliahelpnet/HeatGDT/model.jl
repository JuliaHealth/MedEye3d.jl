module HeatGDTModel

using Lux
using NNlib
using CUDA
using Random
include("heat_diffuse.jl")
using .HeatDiffusion
include("architectures/hed3d.jl")
include("architectures/hed3d_wide.jl")
include("architectures/hed3d_ultra.jl")

export MultimodalEdgeNet, DeepMultimodalEdgeNet3D, HeavyMultimodalEdgeNet3D, HED3D, HED3D_Wide, HED3D_Ultra, HeatGDTFramework, compute_diffusivity

function conv_block(in_c::Int, out_c::Int)
    Chain(
        Conv((3, 3, 3), in_c => out_c, pad=1),
        BatchNorm(out_c, momentum=0.01f0, epsilon=1f-3),
        x -> relu.(x),
        Conv((3, 3, 3), out_c => out_c, pad=1),
        BatchNorm(out_c, momentum=0.01f0, epsilon=1f-3),
        x -> relu.(x)
    )
end

"""
    MultimodalEdgeNet(in_channels=3)

Lightweight prototype 3D Multimodal Edge Detection Network (~30k parameters).
"""
function MultimodalEdgeNet(in_channels::Int=3)
    Chain(
        Conv((3, 3, 3), in_channels => 16, pad=1),
        BatchNorm(16, momentum=0.01f0, epsilon=1f-3),
        x -> relu.(x),
        Conv((3, 3, 3), 16 => 32, pad=1),
        BatchNorm(32, momentum=0.01f0, epsilon=1f-3),
        x -> relu.(x),
        Conv((3, 3, 3), 32 => 16, pad=1),
        BatchNorm(16, momentum=0.01f0, epsilon=1f-3),
        x -> relu.(x),
        Conv((1, 1, 1), 16 => 1),
        x -> sigmoid.(x)
    )
end

"""
    DeepMultimodalEdgeNet3D(in_ch=3)

Medium-capacity 3D Multimodal Edge Detection Network (~1.43M trainable parameters).
Hierarchical feature extraction fusing CT structural anatomy, unscaled PET SUV,
and TotalSegmentator organ atlas priors through multi-scale side outputs.
"""
struct DeepMultimodalEdgeNet3D <: Lux.AbstractLuxContainerLayer{(:e1, :e2, :e3, :d2, :d1, :up2, :up1, :s1, :s2, :s3, :fuse)}
    e1
    e2
    e3
    d2
    d1
    up2
    up1
    s1
    s2
    s3
    fuse
end

function DeepMultimodalEdgeNet3D(in_ch::Int=3)
    # Encoder hierarchy
    e1 = conv_block(in_ch, 32)
    e2 = conv_block(32, 64)
    e3 = conv_block(64, 128)
    
    # Decoder hierarchy
    d2 = conv_block(128 + 64, 64)
    d1 = conv_block(64 + 32, 32)
    
    up2 = Upsample(:trilinear, scale=2)
    up1 = Upsample(:trilinear, scale=2)
    
    # Multi-scale side outputs (deep edge supervision)
    s1 = Conv((1, 1, 1), 32 => 1)
    s2 = Chain(Upsample(:trilinear, scale=2), Conv((1, 1, 1), 64 => 1))
    s3 = Chain(Upsample(:trilinear, scale=4), Conv((1, 1, 1), 128 => 1))
    
    # Final multi-scale fusion head
    fuse = Chain(
        Conv((3, 3, 3), (32 + 3) => 16, pad=1),
        BatchNorm(16, momentum=0.01f0, epsilon=1f-3),
        x -> relu.(x),
        Conv((1, 1, 1), 16 => 1),
        x -> sigmoid.(x)
    )
    return DeepMultimodalEdgeNet3D(e1, e2, e3, d2, d1, up2, up1, s1, s2, s3, fuse)
end

function (m::DeepMultimodalEdgeNet3D)(x, ps, st)
    # x: (64, 64, 64, 3, B)
    # Stage 1
    x1, st_e1 = m.e1(x, ps.e1, st.e1)
    side1, st_s1 = m.s1(x1, ps.s1, st.s1)
    
    # Stage 2
    p1 = NNlib.maxpool(x1, (2, 2, 2))
    x2, st_e2 = m.e2(p1, ps.e2, st.e2)
    side2, st_s2 = m.s2(x2, ps.s2, st.s2)
    
    # Stage 3
    p2 = NNlib.maxpool(x2, (2, 2, 2))
    x3, st_e3 = m.e3(p2, ps.e3, st.e3)
    side3, st_s3 = m.s3(x3, ps.s3, st.s3)
    
    # Decoder 2 (Upsample Stage 3 + Cat Stage 2)
    u2, st_up2 = m.up2(x3, ps.up2, st.up2)
    cat2 = cat(u2, x2, dims=4)
    dec2, st_d2 = m.d2(cat2, ps.d2, st.d2)
    
    # Decoder 1 (Upsample Decoder 2 + Cat Stage 1)
    u1, st_up1 = m.up1(dec2, ps.up1, st.up1)
    cat1 = cat(u1, x1, dims=4)
    dec1, st_d1 = m.d1(cat1, ps.d1, st.d1)
    
    # Multi-scale fusion of all stages
    cat_all = cat(dec1, side1, side2, side3, dims=4)
    edge_out, st_fuse = m.fuse(cat_all, ps.fuse, st.fuse)
    
    st_new = (
        e1=st_e1, e2=st_e2, e3=st_e3,
        d2=st_d2, d1=st_d1,
        up2=st_up2, up1=st_up1,
        s1=st_s1, s2=st_s2, s3=st_s3,
        fuse=st_fuse
    )
    return edge_out, st_new
end

"""
    HeavyMultimodalEdgeNet3D(in_ch=3)

High-capacity 3D Multimodal Edge Detection Network (~24M - 26M trainable parameters),
matching standard 3D HELPNet capacity. 5 encoder stages [32, 64, 128, 256, 512]
with multi-scale side edge supervision and fusion head.
"""
struct HeavyMultimodalEdgeNet3D <: Lux.AbstractLuxContainerLayer{(:e1, :e2, :e3, :e4, :e5, :d4, :d3, :d2, :d1, :up4, :up3, :up2, :up1, :s1, :s2, :s3, :s4, :fuse)}
    e1
    e2
    e3
    e4
    e5
    d4
    d3
    d2
    d1
    up4
    up3
    up2
    up1
    s1
    s2
    s3
    s4
    fuse
end

function HeavyMultimodalEdgeNet3D(in_ch::Int=3)
    # 5 Encoder stages
    e1 = conv_block(in_ch, 32)
    e2 = conv_block(32, 64)
    e3 = conv_block(64, 128)
    e4 = conv_block(128, 256)
    e5 = conv_block(256, 512)
    
    # 4 Decoder stages
    d4 = conv_block(512 + 256, 256)
    d3 = conv_block(256 + 128, 128)
    d2 = conv_block(128 + 64, 64)
    d1 = conv_block(64 + 32, 32)
    
    up4 = Upsample(:trilinear, scale=2)
    up3 = Upsample(:trilinear, scale=2)
    up2 = Upsample(:trilinear, scale=2)
    up1 = Upsample(:trilinear, scale=2)
    
    # Side outputs
    s1 = Conv((1, 1, 1), 32 => 1)
    s2 = Chain(Upsample(:trilinear, scale=2), Conv((1, 1, 1), 64 => 1))
    s3 = Chain(Upsample(:trilinear, scale=4), Conv((1, 1, 1), 128 => 1))
    s4 = Chain(Upsample(:trilinear, scale=8), Conv((1, 1, 1), 256 => 1))
    
    # Final fusion
    fuse = Chain(
        Conv((3, 3, 3), (32 + 4) => 16, pad=1),
        BatchNorm(16, momentum=0.01f0, epsilon=1f-3),
        x -> relu.(x),
        Conv((1, 1, 1), 16 => 1),
        x -> sigmoid.(x)
    )
    return HeavyMultimodalEdgeNet3D(e1, e2, e3, e4, e5, d4, d3, d2, d1, up4, up3, up2, up1, s1, s2, s3, s4, fuse)
end

function (m::HeavyMultimodalEdgeNet3D)(x, ps, st)
    # x: (64, 64, 64, 3, B)
    # Stage 1
    x1, st_e1 = m.e1(x, ps.e1, st.e1)
    side1, st_s1 = m.s1(x1, ps.s1, st.s1)
    
    # Stage 2
    p1 = NNlib.maxpool(x1, (2, 2, 2))
    x2, st_e2 = m.e2(p1, ps.e2, st.e2)
    side2, st_s2 = m.s2(x2, ps.s2, st.s2)
    
    # Stage 3
    p2 = NNlib.maxpool(x2, (2, 2, 2))
    x3, st_e3 = m.e3(p2, ps.e3, st.e3)
    side3, st_s3 = m.s3(x3, ps.s3, st.s3)
    
    # Stage 4
    p3 = NNlib.maxpool(x3, (2, 2, 2))
    x4, st_e4 = m.e4(p3, ps.e4, st.e4)
    side4, st_s4 = m.s4(x4, ps.s4, st.s4)
    
    # Stage 5 (Bottleneck)
    p4 = NNlib.maxpool(x4, (2, 2, 2))
    x5, st_e5 = m.e5(p4, ps.e5, st.e5)
    
    # Decoder 4
    u4, st_up4 = m.up4(x5, ps.up4, st.up4)
    dec4, st_d4 = m.d4(cat(u4, x4, dims=4), ps.d4, st.d4)
    
    # Decoder 3
    u3, st_up3 = m.up3(dec4, ps.up3, st.up3)
    dec3, st_d3 = m.d3(cat(u3, x3, dims=4), ps.d3, st.d3)
    
    # Decoder 2
    u2, st_up2 = m.up2(dec3, ps.up2, st.up2)
    dec2, st_d2 = m.d2(cat(u2, x2, dims=4), ps.d2, st.d2)
    
    # Decoder 1
    u1, st_up1 = m.up1(dec2, ps.up1, st.up1)
    dec1, st_d1 = m.d1(cat(u1, x1, dims=4), ps.d1, st.d1)
    
    # Multi-scale fusion
    cat_all = cat(dec1, side1, side2, side3, side4, dims=4)
    edge_out, st_fuse = m.fuse(cat_all, ps.fuse, st.fuse)
    
    st_new = (
        e1=st_e1, e2=st_e2, e3=st_e3, e4=st_e4, e5=st_e5,
        d4=st_d4, d3=st_d3, d2=st_d2, d1=st_d1,
        up4=st_up4, up3=st_up3, up2=st_up2, up1=st_up1,
        s1=st_s1, s2=st_s2, s3=st_s3, s4=st_s4,
        fuse=st_fuse
    )
    return edge_out, st_new
end

"""
    compute_diffusivity(edge_map, epsilon_D=0.01f0, gamma=5.0f0)

Converts edge probability map E ∈ [0, 1] into a diffusion conductance field D ∈ [epsilon_D, 1.0]:
D(x) = epsilon_D + (1.0 - epsilon_D) * exp(-gamma * E(x))
"""
function compute_diffusivity(edge_map, epsilon_D::Float32=0.01f0, gamma::Float32=5.0f0)
    e = edge_map
    return epsilon_D .+ (1.0f0 - epsilon_D) .* exp.(-gamma .* e)
end

struct HeatGDTFramework{M} <: Lux.AbstractLuxContainerLayer{(:edge_net,)}
    edge_net::M
    K::Int
    dt::Float32
    theta::Float32
    tau::Float32
end

function HeatGDTFramework(; in_channels=3, edge_type=:deep, K=20, dt=0.12f0, theta=0.012f0, tau=0.03f0)
    edge_net = if edge_type == :heavy
        HeavyMultimodalEdgeNet3D(in_channels)
    elseif edge_type == :prototype
        MultimodalEdgeNet(in_channels)
    elseif edge_type == :hed3d
        HED3D(in_channels)
    elseif edge_type == :hed3d_wide
        HED3D_Wide(in_channels)
    elseif edge_type == :hed3d_ultra
        HED3D_Ultra(in_channels)
    else
        DeepMultimodalEdgeNet3D(in_channels)
    end
    return HeatGDTFramework(edge_net, K, dt, theta, tau)
end

function (m::HeatGDTFramework)((x, seed_mask), ps, st)
    # x: (X, Y, Z, 3, B) [Ch 1: CT, Ch 2: PET, Ch 3: TotalSeg]
    # seed_mask: (X, Y, Z, B) with 1.0 at seed voxel, 0 elsewhere
    
    edge_map, st_edge = m.edge_net(x, ps.edge_net, st.edge_net)
    D = compute_diffusivity(edge_map)
    
    # Differentiable heat diffusion
    u = heat_diffusion(seed_mask, D, m.K, m.dt)
    
    # Soft lesion mask extraction
    pred_mask = extract_mask(u, m.theta, m.tau)
    
    return (mask=pred_mask, edges=edge_map, heat=u, D=D), (edge_net=st_edge,)
end

end # module
