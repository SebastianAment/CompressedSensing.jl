module TestBackward
using CompressedSensing
using CompressedSensing: gaussian_data, br, fbr, lace, FBR, BR, backward_step!, perturb, correlated_data

using Test
using LinearAlgebra
using SparseArrays

# set up data
n, k = 64, 16
A, x, b = gaussian_data(n = n, m = n, k = k)
δ = 1e-2
y = perturb(b, δ/2)

@testset "Backward Regression" begin
    xbr = br(A, y, sparsity = k) # k-sparse approximation
    @test x.nzind == xbr.nzind # support recovery
    @test isapprox(x, xbr, atol = 2δ) # support recovery

    xbr = br(A, y, max_residual = δ) # δ-accurate approximation
    @test x.nzind == xbr.nzind
    @test isapprox(x, xbr, atol = 2δ) # support recovery

    xbr = br(A, y, max_increase = δ) # approximation with no marginal norm increase above δ
    @test x.nzind == xbr.nzind
    @test isapprox(x, xbr, atol = 2δ) # support recovery
end

@testset "LACE" begin
    xlace = lace(A, y, sparsity = k) # k-sparse approximation
    @test x.nzind == xlace.nzind # support recovery
    @test isapprox(x, xlace, atol = 2δ) # support recovery

    xlace = lace(A, y, max_residual = δ) # δ-accurate approximation
    @test x.nzind == xlace.nzind
    @test isapprox(x, xlace, atol = 2δ) # support recovery

    xlace = lace(A, y, max_increase = δ) # approximation with no marginal norm increase above δ
    @test x.nzind == xlace.nzind
    @test isapprox(x, xlace, atol = 2δ) # support recovery
end

# based on low rank updates to normal equations
@testset "Fast Backward Regression" begin
    xbr = fbr(A, y, sparsity = k) # k-sparse approximation
    @test x.nzind == xbr.nzind # support recovery
    @test isapprox(x, xbr, atol = 2δ) # support recovery

    xbr = fbr(A, y, max_residual = δ) # δ-accurate approximation
    @test x.nzind == xbr.nzind
    @test isapprox(x, xbr, atol = 2δ) # support recovery

    xbr = fbr(A, y, max_increase = δ) # approximation with no marginal norm increase above δ
    @test x.nzind == xbr.nzind
    @test isapprox(x, xbr, atol = 2δ) # support recovery
end

# using BenchmarkTools
# @btime br($A, $y, sparsity = $k)
# @btime fbr($A, $y, sparsity = $k)

# function residual_magnitude(n::Int)
#     A = randn(n, n-1)
#     a = randn(n)
#     norm(a-A\a)
# end

end # TestBackward
