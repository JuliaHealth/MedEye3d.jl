struct ExtractMaskPC
    a::Float32; b::Float32; c::Int32; d::Int32
    e::Int32; f::Int32; g::Int32; h::Int32
    i::Int32; j::Int32; k::Int32; l::Int32
end
try
    ExtractMaskPC(1f0, 2f0, Int32(3), Int32(4))
    println("Success")
catch e
    println("Error: ", e)
end
