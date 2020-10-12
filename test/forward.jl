module TestForward
using CompressedSensing
using CompressedSensing: sparse_data, fr, foba, perturb

using Test
using LinearAlgebra
n, m, k = 32, 64, 4
A, x, b = sparse_data(n = n, m = m, k = k) # needs to be determined
δ = 1e-2
y = perturb(b, δ)

@testset "forward regression" begin
    # noiseless
    xfr = fr(A, b, sparsity = k)
    @test xfr.nzind == x.nzind
    @test xfr.nzval ≈ x.nzval

    # with small perturbation
    xfr = fr(A, y, sparsity = k)
    @test xfr.nzind == x.nzind # support recovery
    @test isapprox(xfr.nzval, x.nzval, atol = 2δ)

end

@testset "FoBa" begin
    xfr = foba(A, y, δ)
    @test xfr.nzind == x.nzind # support recovery
    @test isapprox(xfr.nzval, x.nzval, atol = 2δ)
end

end # TestForward
