- Compound loss with DTM boundary supervision
"""
struct HeatGDTFrameworkV2{M} <: Lux.AbstractLuxContainerLayer{(:edge_net,)}
    edge_net::M
    K::Int
    dt::Float32
    theta::Float32
    tau::Float32
    pde_type::Symbol      # :pure_diffusion, :reaction_diffusion, :advection_diffusion, :full_rda
    edge_type::Symbol     # :deep, :heavy, :hed3d, :dexined3d, :dframenet, :hybrid25d
end

function create_edge_net(edge_type::Symbol, in_ch::Int=3)
    if edge_type == :hed3d
        return HED3D(in_ch)
    elseif edge_type == :dexined3d
        return DexiNed3D(in_ch)
    elseif edge_type == :dframenet
        return DFrameNet(in_ch)
    elseif edge_type == :hybrid25d
        return Hybrid25D3D(in_ch)
    elseif edge_type == :heavy
        return HeatGDTModel.HeavyMultimodalEdgeNet3D(in_ch)
    elseif edge_type == :deep
        return HeatGDTModel.DeepMultimodalEdgeNet3D(in_ch)
    else
        return HeatGDTModel.MultimodalEdgeNet(in_ch)
    end
end

function HeatGDTFrameworkV2(;
    in_channels=3,
    edge_type::Symbol=:deep,
    pde_type::Symbol=:pure_diffusion,
    K=20, dt=0.12f0, theta=0.012f0, tau=0.03f0
)
    edge_net = create_edge_net(edge_type, in_channels)
    return HeatGDTFrameworkV2(edge_net, K, dt, theta, tau, pde_type, edge_type)
end

function (m::HeatGDTFrameworkV2)((x, seed_mask), ps, st)
    # x: (X, Y, Z, 3, B), seed_mask: (X, Y, Z, B)
    net_out, st_edge = m.edge_net(x, ps.edge_net, st.edge_net)

    # Handle DFrameNet (returns NamedTuple with edges + direction)
    if m.edge_type == :dframenet
        edge_map = net_out.edges
        direction_field = net_out.direction  # (X, Y, Z, 3, B)
    else
        edge_map = net_out
        direction_field = nothing
    end

    # Compute diffusivity from edge map
    D = HeatGDTModel.compute_diffusivity(edge_map)

    # PDE solve based on type
    if m.pde_type == :pure_diffusion
        u = HeatDiffusion.heat_diffusion(seed_mask, D, m.K, m.dt)
    elseif m.pde_type == :reaction_diffusion
        # Use edge map as absorption field β
        beta = dropdims(edge_map, dims=4) .* 2.0f0  # scale edges → absorption
        v = Zygote.ignore_derivatives() do
            CUDA.zeros(Float32, size(D, 1), size(D, 2), size(D, 3), 3, size(D, 4))
        end
        u = HeatDiffusionV2.heat_diffusion_v2(seed_mask, D, beta, v, m.K, m.dt)
    elseif m.pde_type == :advection_diffusion
        beta = Zygote.ignore_derivatives() do
            CUDA.zeros(Float32, size(D)...)
        end
        if direction_field !== nothing
            u = HeatDiffusionV2.heat_diffusion_v2(seed_mask, D, beta, direction_field, m.K, m.dt)
        else
            v = Zygote.ignore_derivatives() do
                CUDA.zeros(Float32, size(D, 1), size(D, 2), size(D, 3), 3, size(D, 4))
            end
            u = HeatDiffusionV2.heat_diffusion_v2(seed_mask, D, beta, v, m.K, m.dt)
        end
    elseif m.pde_type == :full_rda
        beta = dropdims(edge_map, dims=4) .* 2.0f0
        if direction_field !== nothing
            u = HeatDiffusionV2.heat_diffusion_v2(seed_mask, D, beta, direction_field, m.K, m.dt)
        else
