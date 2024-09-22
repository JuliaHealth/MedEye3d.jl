using TensorBoardLogger: TBLogger, log_value, log_images
using Lux
using CUDA,Random,Optimisers,LuxCUDA

CUDA.allowscalar(true)

# ## Model Definition

function residual_block(in_channels::Int, out_channels::Int)
    return Parallel(+,
        in_channels == out_channels ? NoOpLayer() :
        Conv((1, 1, 1), in_channels => out_channels; pad=SamePad()),
        Chain(BatchNorm(in_channels; affine=false),
            Conv((3, 3, 3), in_channels => out_channels, swish; pad=SamePad()),
            Conv((3, 3, 3), out_channels => out_channels; pad=SamePad()));
        name="ResidualBlock(in_chs=$in_channels, out_chs=$out_channels)")
end

function downsample_block(in_channels::Int, out_channels::Int, block_depth::Int)
    return @compact(;
        name="DownsampleBlock(in_chs=$in_channels, out_chs=$out_channels, block_depth=$block_depth)",
        residual_blocks=Tuple(residual_block(
                                  ifelse(i == 1, in_channels, out_channels), out_channels)
        for i in 1:block_depth),
        meanpool=MeanPool((2, 2, 2)), block_depth) do x
        skips = (x,)
        for i in 1:block_depth
            skips = (skips..., residual_blocks[i](last(skips)))
        end
        y = meanpool(last(skips))
        @return y, skips
    end
end

function upsample_block(in_channels::Int, out_channels::Int, block_depth::Int)
    return @compact(;
        name="UpsampleBlock(in_chs=$in_channels, out_chs=$out_channels, block_depth=$block_depth)",
        residual_blocks=Tuple(residual_block(
                                  ifelse(
                                      i == 1, in_channels + out_channels, out_channels * 2),
                                  out_channels) for i in 1:block_depth),
        upsample=Upsample(:bilinear; scale=2), block_depth) do x_skips
        x, skips = x_skips
        x = upsample(x)
        for i in 1:block_depth
            x = residual_blocks[i](cat(x, skips[end - i + 1]; dims=Val(4)))
        end
        @return x
    end
end

function unet_model(image_size::Tuple{Int, Int, Int}; channels=[32, 64, 96, 128],
        block_depth=2)
    upsample = Upsample(:nearest; size=image_size)
    conv_in = Conv((1, 1, 1), 3 => channels[1])
    conv_out = Conv((1, 1, 1), channels[1] => 3; init_weight=Lux.zeros32)

    down_blocks = [downsample_block(
                       i == 1 ? channels[1] : channels[i - 1], channels[i], block_depth)
                   for i in 1:(length(channels) - 1)]
    residual_blocks = Chain([residual_block(
                                 ifelse(i == 1, channels[end - 1], channels[end]),
                                 channels[end]) for i in 1:block_depth]...)

    reverse!(channels)
    up_blocks = [upsample_block(in_chs, out_chs, block_depth)
                 for (in_chs, out_chs) in zip(channels[1:(end - 1)], channels[2:end])]

    #! format: off
    return @compact(;
        upsample, conv_in, conv_out, down_blocks, residual_blocks, up_blocks,
        num_blocks=(length(channels) - 1)) do x::AbstractArray{<:Real, 5}
    #! format: on
        x = conv_in(x)
        skips_at_each_stage = ()
        for i in 1:num_blocks
            x, skips = down_blocks[i](x)
            skips_at_each_stage = (skips_at_each_stage..., skips)
        end
        x = residual_blocks(x)
        for i in 1:num_blocks
            x = up_blocks[i]((x, skips_at_each_stage[end - i + 1]))
        end
        @return conv_out(x)
    end
end

function infer_model(tstate_glob, model, imagee)
    y_pred, st = Lux.apply(model, CuArray(imagee), tstate_glob.parameters, tstate_glob.states)
    return y_pred, st
end

# Test case
function test_unet_model()
    rng = Random.default_rng()
    opt = Optimisers.Lion(0.000002)

    image_size = (64, 64, 64)
    model = unet_model(image_size)
    tstate = Lux.Experimental.TrainState(rng, model, opt)

    input = CuArray(rand(Float32, 64, 64, 64, 3, 2))  # (x, y, z, channel, batch)
    y_pred, st = infer_model(tstate, model, input)
    println("Output size: ", size(y_pred))
end

test_unet_model()