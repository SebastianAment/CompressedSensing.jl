module TestForward
using CompressedSensing
using CompressedSensing: sparse_data, fr, foba

using Test
using LinearAlgebra
@testset "forward regression" begin
    n, m, k = 32, 64, 4
    δ = .01
    A, x, b = sparse_data(n = n, m = m, k = k, min_x = δ) # needs to be determined

    xfr = fr(A, b, sparsity = k)
    # noiseless
    @test xfr.nzind == x.nzind
    @test xfr.nzval ≈ x.nzval

    # small perturbation
    ε = randn(n)
    ε .*= δ/norm(ε)
    y = b + ε
    xfr = fr(A, y, sparsity = k)
    @test xfr.nzind == x.nzind # support recovery
    @test isapprox(xfr.nzval, x.nzval, atol = 2δ)

    xfr = foba(A, y)
end

end # TestForward
