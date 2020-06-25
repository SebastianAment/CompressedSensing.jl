module TestBasisPursuit
using Test
using LinearAlgebra
using SparseArrays
using CompressedSensing: bp, bp_candes, bp_ard, bpd, bpd_candes, bpd_ard, sparse_data

@testset "Basis Pursuit" begin
    # equality constrained l1 minimization
    n, m = 32, 64
    k = 6
    A, x, b = sparse_data(n = n, m = m, k = k, rescaled = true)
    xl = bp(A, b)
    @test xl.nzind == x.nzind

    xc = bp_candes(A, b)
    @test xc.nzind == x.nzind

    xard = bp_ard(A, b)
    @test xard.nzind == x.nzind
end

@testset "Basis Pursuit Denoising" begin
    # equality constrained l1 minimization
    n, m = 32, 64
    k = 6
    δ = 1e-2
    min_x = 2e-2 # above noise-level to make bp work
    A, x, b = sparse_data(n = n, m = m, k = k, min_x = min_x, rescaled = true)
    e = randn(n)
    e *= δ/2norm(e) # create δ perturbation from b
    b += e

    xl = bpd(A, b, δ) # still has spurious coefficients above an order of magnitudes below perturbation level
    droptol!(xl, 1e-2) # eh
    @test xl.nzind == x.nzind

    xc = bpd_candes(A, b, δ, maxiter = 3)
    # println("candes")
    droptol!(xc, 1e-6)
    @test xc.nzind == x.nzind

    xard = bpd_ard(A, b, δ, maxiter = 16)
    # println("ard")
    droptol!(xard, 1e-6)
    @test xard.nzind == x.nzind
end

end
