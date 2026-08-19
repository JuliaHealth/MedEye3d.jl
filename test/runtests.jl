using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Test

@testset "MedEye3d.jl" begin
    @testset "Unit Tests" begin
        include("test_all.jl")
    end
    @testset "Spacing" begin
        include("test_spacing.jl")
    end
    @testset "Vertices" begin
        include("test_vertices.jl")
    end
    @testset "Bone Subsegmentation" begin
        include("test_bone_subseg.jl")
    end
    @testset "Inference Client" begin
        include("test_inference.jl")
    end
end
