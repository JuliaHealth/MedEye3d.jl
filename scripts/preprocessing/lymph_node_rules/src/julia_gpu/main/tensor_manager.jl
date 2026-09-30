using CUDA
using KernelAbstractions

mutable struct TensorManager
    backend::Backend
    channels::Dict{Int, AbstractArray}
    size_x::Int
    size_y::Int
    size_z::Int
    max_channels::Int
end

function init_tensor_manager(backend::Backend, size_x::Int, size_y::Int, size_z::Int, max_channels::Int)
    println("TensorManager: Initializing with max $max_channels channels")
    
    channels = Dict{Int, AbstractArray}()
    
    return TensorManager(backend, channels, size_x, size_y, size_z, max_channels)
end

function get_channel_view(tm::TensorManager, ch::Int)
    if !haskey(tm.channels, ch)
        tm.channels[ch] = KernelAbstractions.zeros(tm.backend, UInt8, tm.size_x, tm.size_y, tm.size_z)
    end
    return tm.channels[ch]
end

function get_empty_view(tm::TensorManager)
    return get_channel_view(tm, 0)
end

function allocate_channel!(tm::TensorManager)
    # No longer needed, python manages IDs
    return 1
end

function free_channel!(tm::TensorManager, ch::Int)
    # Free memory immediately to avoid swapping
    if haskey(tm.channels, ch)
        delete!(tm.channels, ch)
    end
end

function load_mask_to_channel!(tm::TensorManager, ch::Int, mask::AbstractArray)
    d_view = get_channel_view(tm, ch)
    copyto!(d_view, mask)
end

function save_channel_to_mask(tm::TensorManager, ch::Int)
    d_view = get_channel_view(tm, ch)
    mask = zeros(UInt8, tm.size_x, tm.size_y, tm.size_z)
    copyto!(mask, d_view)
    return mask
end
