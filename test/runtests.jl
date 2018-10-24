using Test
using SurfaceTools
using StaticArrays
using LinearAlgebra

@testset "sphere" begin
    s(p) = @SVector [p[1], p[2], sqrt(1 - p ⋅ p)]
    @test curvature(s, @SVector [0., 0.]) ≈ -1
    @test gram_det(s, @SVector [0., 0.]) ≈ 1
end
