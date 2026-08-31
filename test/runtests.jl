using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Test

@testset "MedEye3d.jl" begin
    @testset "Unit Tests" begin
        include("test_all.jl")
    end
    @testset "Vulkan Backend" begin
        include("test_vulkan_backend.jl")
    end
end
