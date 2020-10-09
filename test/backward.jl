module TestBackward
using CompressedSensing
using CompressedSensing: sparse_data, br, fbr, lace, FBR, BR, backward_step!

using Test
using LinearAlgebra
using SparseArrays

# set up data
n, k = 128, 16
δ = 1e-2
A, x, b = sparse_data(n = n, m = n, k = k, min_x = √2δ) # needs to be determined
ε = randn(n)
ε .*= δ/2norm(ε)
y = b + ε

@testset "backward regression" begin
    xbr = br(A, y, sparsity = k) # k-sparse approximation
    @test x.nzind == xbr.nzind # support recovery
    xbr = br(A, y, max_residual = δ) # δ-accurate approximation
    @test x.nzind == xbr.nzind
    xbr = br(A, y, max_increase = δ) # approximation with no marginal norm increase above δ
    @test x.nzind == xbr.nzind
end

@testset "lace" begin
    xlace = lace(A, y, Inf, k) # k-sparse approximation
    @test x.nzind == xlace.nzind # support recovery
    # xbr = br(A, y, max_residual = δ) # δ-accurate approximation
    # @test x.nzind == xbr.nzind
    # xbr = br(A, y, max_increase = δ) # approximation with no marginal norm increase above δ
    # @test x.nzind == xbr.nzind
end

@testset "fast backward regression" begin
    xbr = fbr(A, y, sparsity = k) # k-sparse approximation
    @test x.nzind == xbr.nzind # support recovery
    xbr = fbr(A, y, max_residual = δ) # δ-accurate approximation
    @test x.nzind == xbr.nzind
    xbr = fbr(A, y, max_increase = δ) # approximation with no marginal norm increase above δ
    @test x.nzind == xbr.nzind
end
#
using BenchmarkTools
@btime br($A, $y, sparsity = $k)
@btime fbr($A, $y, sparsity = $k)

function residual_magnitude(n::Int)
    A = randn(n, n-1)
    a = randn(n)
    norm(a-A\a)
end

end # TestBackward
