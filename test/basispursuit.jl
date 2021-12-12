module TestBasisPursuit
using Test
using LinearAlgebra
using SparseArrays
using CompressedSensing: bp, bp_candes, bp_ard, bpd, bpd_candes, bpd_ard, sparse_data, perturb

n, m = 32, 48
k = 3
A, x, b = sparse_data(n = n, m = m, k = k, rescaled = true)
δ = 1e-2
y = perturb(b, δ/2)

@testset "Basis Pursuit" begin
    # equality constrained l1 minimization
    xl = bp(A, b)
    @test xl.nzind == x.nzind

    xc = bp_candes(A, b)
    @test xc.nzind == x.nzind

    xard = bp_ard(A, b)
    @test xard.nzind == x.nzind
end

@testset "Basis Pursuit Denoising" begin
    xl = bpd(A, y, δ)
    droptol!(xl, 1e-2) # sometimes has spurious coefficients above perturbation level
    @test xl.nzind == x.nzind

    xc = bpd_candes(A, y, δ, maxiter = 3)
    droptol!(xc, 1e-6)
    @test xc.nzind == x.nzind

    xard = bpd_ard(A, y, δ, maxiter = 16)
    droptol!(xard, 1e-6)
    @test xard.nzind == x.nzind
end

using CompressedSensing: ista, fista
@testset "ISTA" begin
    λ = δ/10
    xista = ista(A, y, λ, maxiter = 1024, stepsize = 1e-1)
    # droptol!(xista, δ)
    # @test xista.nzind == x.nzind
    # @test xista.nzind ⊆ x.nzind
    @test norm(A*xista - y) < δ
    # TODO: FISTA
end

end
