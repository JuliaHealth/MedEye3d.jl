using Lux
using NNlib
using Random

function conv_block_3d(in_ch::Int, out_ch::Int)
    Chain(
        Conv((3, 3, 3), in_ch => out_ch, pad=1),
        x -> relu.(x),
        BatchNorm(out_ch, momentum=0.01f0, epsilon=1f-3),
        Conv((3, 3, 3), out_ch => out_ch, pad=1),
        x -> relu.(x),
        BatchNorm(out_ch, momentum=0.01f0, epsilon=1f-3)
    )
end

function up_conv_3d(in_ch::Int, out_ch::Int)
    Chain(
        Upsample(:trilinear, scale=2, align_corners=true),
        Conv((3, 3, 3), in_ch => out_ch, pad=1),
        x -> relu.(x),
        BatchNorm(out_ch, momentum=0.01f0, epsilon=1f-3)
    )
end

struct HELPNet_Lesion <: Lux.AbstractLuxContainerLayer{(:e1, :e2, :e3, :e4, :e5, :up4, :up_conv4, :up3, :up_conv3, :up2, :up_conv2, :up1, :up_conv1, :conv, :norm)}
    e1
    e2
    e3
    e4
    e5
    up4
    up_conv4
    up3
    up_conv3
    up2
    up_conv2
    up1
    up_conv1
    conv
    norm
end

function HELPNet_Lesion(in_ch=3, out_ch=2)
    filters = [32, 64, 128, 256, 512]
    
    e1 = conv_block_3d(in_ch, filters[1])
    e2 = Chain(MaxPool((2, 2, 2)), conv_block_3d(filters[1], filters[2]))
    e3 = Chain(MaxPool((2, 2, 2)), conv_block_3d(filters[2], filters[3]))
    e4 = Chain(MaxPool((2, 2, 2)), conv_block_3d(filters[3], filters[4]))
    e5 = Chain(MaxPool((2, 2, 2)), conv_block_3d(filters[4], filters[5]))
    
    up4 = up_conv_3d(filters[5], filters[4])
    up_conv4 = conv_block_3d(filters[5], filters[4])
    
    up3 = up_conv_3d(filters[4], filters[3])
    up_conv3 = conv_block_3d(filters[4], filters[3])
    
    up2 = up_conv_3d(filters[3], filters[2])
    up_conv2 = conv_block_3d(filters[3], filters[2])
    
    up1 = up_conv_3d(filters[2], filters[1])
    up_conv1 = conv_block_3d(filters[2], filters[1])
    
    conv = Conv((1, 1, 1), filters[1] => out_ch)
    norm = BatchNorm(out_ch, momentum=0.01f0, epsilon=1f-3)
    
    HELPNet_Lesion(e1, e2, e3, e4, e5, up4, up_conv4, up3, up_conv3, up2, up_conv2, up1, up_conv1, conv, norm)
end

function (m::HELPNet_Lesion)(x, ps, st)
    x1, st_e1 = m.e1(x, ps.e1, st.e1)
    x2, st_e2 = m.e2(x1, ps.e2, st.e2)
    x3, st_e3 = m.e3(x2, ps.e3, st.e3)
    x4, st_e4 = m.e4(x3, ps.e4, st.e4)
    x5, st_e5 = m.e5(x4, ps.e5, st.e5)
    
    d4, st_up4 = m.up4(x5, ps.up4, st.up4)
    # PyTorch: torch.cat((d4, e4), dim=1)
    # In Julia, spatial dims are 1..N, channels is N+1, batch is N+2. For 3D, channels is dim 4.
    c4 = cat(d4, x4; dims=4)
    d4_out, st_up_conv4 = m.up_conv4(c4, ps.up_conv4, st.up_conv4)
    
    d3, st_up3 = m.up3(d4_out, ps.up3, st.up3)
    c3 = cat(d3, x3; dims=4)
    d3_out, st_up_conv3 = m.up_conv3(c3, ps.up_conv3, st.up_conv3)
    
    d2, st_up2 = m.up2(d3_out, ps.up2, st.up2)
    c2 = cat(d2, x2; dims=4)
    d2_out, st_up_conv2 = m.up_conv2(c2, ps.up_conv2, st.up_conv2)
    
    d1, st_up1 = m.up1(d2_out, ps.up1, st.up1)
    c1 = cat(d1, x1; dims=4)
    d1_out, st_up_conv1 = m.up_conv1(c1, ps.up_conv1, st.up_conv1)
    
    d0, st_conv = m.conv(d1_out, ps.conv, st.conv)
    norm_out, st_norm = m.norm(d0, ps.norm, st.norm)
    
    # Use softmax over channels (dim 4)
    out = NNlib.softmax(norm_out, dims=4)
    
    st_new = (
        e1=st_e1, e2=st_e2, e3=st_e3, e4=st_e4, e5=st_e5,
        up4=st_up4, up_conv4=st_up_conv4, up3=st_up3, up_conv3=st_up_conv3,
        up2=st_up2, up_conv2=st_up_conv2, up1=st_up1, up_conv1=st_up_conv1,
        conv=st_conv, norm=st_norm
    )
    
    return out, st_new
end
