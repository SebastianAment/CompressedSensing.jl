module TestSBL
using Test
using LinearAlgebra
using CompressedSensing: sparse_data, sbl, SBL, fsbl, rmps, update!, rmp, foba
using SparseArrays

n, m, k = 64, 128, 6
σ = 1e-2
Σ = σ^2*I(n)
A, x, b = sparse_data(n = n, m = m, k = k)
ε = randn(n)
ε ./= norm(ε)
y = @. b + σ/2*ε

@testset "Sparse Bayesian Learning" begin
    xsbl = sbl(A, y, σ)
    tol = σ
    # xsbl = droptol!(sparse(xsbl), tol)
    @test findall(abs.(xsbl) .> tol) == x.nzind
    @test isapprox(A*xsbl, b, atol = σ)

    # fast sparse bayesian learning
    xfsbl = fsbl(A, y, σ)
    @test findall(abs.(xfsbl) .> tol) == x.nzind
    @test isapprox(A*xfsbl, b, atol = σ)

    # relevance matching pursuit
    xrmp = rmps(A, y, σ)
    @test findall(abs.(xrmp) .> tol) == x.nzind
    @test isapprox(A*xrmp, b, atol = σ)

    # zero noise limit of relevance matching pursuit
    xrmp = rmp(A, y, σ)
    @test xrmp.nzind == x.nzind
    @test isapprox(A*xrmp, b, atol = σ)

end

bench = false
using BenchmarkTools
if bench
    # @btime sbl($A, $b, $σ)
    @btime rmps($A, $b, $σ)
    @btime fsbl($A, $b, $σ)
    @btime rmp($A, $y, $σ)
    @btime foba($A, $y, $σ)

end
end # TestSBL
