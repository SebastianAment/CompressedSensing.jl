module TestSBL
using Test
using LinearAlgebra
using CompressedSensing: sparse_data, SBL, greedy_sbl, rp
using SparseArrays

@testset "Sparse Bayesian Learning" begin
    n, m, k = 32, 64, 3
    σ = 1e-2
    Σ = σ^2*I(n)
    A, x, b = sparse_data(n = n, m = m, k = k)
    @. b += σ * randn()

    sbl = SBL(A, b, Σ)
    xsbl = zeros(m)
    for i in 1:128
        sbl(xsbl)
    end
    tol = 10σ
    # xsbl = droptol!(sparse(xsbl), tol)
    @test findall(abs.(xsbl) .> tol) == x.nzind

    # greedy sparse bayesian learning
    xgsbl = greedy_sbl(A, b, σ)

    @test findall(abs.(xgsbl) .> tol) == x.nzind
    @test isapprox(A*xgsbl, b, atol = 5σ)

    xrp = rp(A, b, σ)
    @test findall(abs.(xrp) .> tol) == x.nzind
end

end # TestSBL
