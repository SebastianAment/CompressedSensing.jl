module TestSBL
using Test
using LinearAlgebra
using CompressedSensing: sparse_data, sbl, SBL, fsbl, rmps, update!, rmp, foba, perturb
using SparseArrays

n, m, k = 64, 128, 6
A, x, b = sparse_data(n = n, m = m, k = k)
σ = 1e-2
y = perturb(b, σ/2)

@testset "Sparse Bayesian Learning" begin
    xsbl = sbl(A, y, σ^2) # need to pass variance or covariance of noise
    tol = σ
    # xsbl = droptol!(sparse(xsbl), tol)
    @test findall(abs.(xsbl) .> tol) == x.nzind
    @test isapprox(A*xsbl, b, atol = σ)

    # fast sparse bayesian learning
    xfsbl = fsbl(A, y, σ^2)
    @test findall(abs.(xfsbl) .> tol) == x.nzind
    @test isapprox(A*xfsbl, b, atol = σ)

    # relevance matching pursuit
    xrmps = rmps(A, y, σ^2)
    @test findall(abs.(xrmps) .> tol) == x.nzind
    @test isapprox(A*xrmps, b, atol = σ)

    # rmp with noise variance optimization
    σ_init = σ
    xrmps, σ²_opt = rmps(A, y, Val(true), σ_init^2)
    @test xrmps isa SparseVector
    @test σ²_opt isa Real
    @test norm(A*xrmps-y) < 5sqrt(σ²_opt) * size(A, 1)

    # rmp with noise variance optimization with prior over σ
    σ_init = σ
    xrmps, σ²_opt = rmps(A, y, Val(true), σ_init^2, 1, σ_init^2)
    @test norm(A*xrmps-y) < 5sqrt(σ²_opt) * size(A, 1)
    @test isapprox(σ²_opt, σ^2, rtol = 2) # noise variance is approximately recovered

    # zero noise limit of relevance matching pursuit
    xrmp = rmp(A, y, σ)
    @test xrmp.nzind == x.nzind
    @test isapprox(A*xrmp, b, atol = σ)

end

# bench = false
# using BenchmarkTools
# if bench
#     @btime sbl($A, $b, $(σ^2))
#     @btime rmps($A, $b, $(σ^2))
#     @btime fsbl($A, $b, $(σ^2))
#     @btime rmp($A, $y, $(σ^2))
#     @btime foba($A, $y, $(σ^2))
# end

end # TestSBL
