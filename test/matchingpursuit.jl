module TestMatchingPursuit
using Test
using LinearAlgebra
using CompressedSensing: MP, mp, OMP, omp, SP, sp, RMP, rmp, sparse_data,
                        oomp, ompr
using SparseArrays

# TODO: pool tests of similar algorithms, e.g.: omp, oomp, ompr

@testset "Matching Pursuit" begin
    n, m, k = 32, 32, 3
    σ = 0.
    A, x, b = sparse_data(n = n, m = m, k = k, rescaled = true)

    xmp = mp(A, b, 3k) # could give it more iterations, might re-optimize old atom
    # noiseless
    # @test xmp.nzind == x.nzind
    @test isapprox(A*xmp, b, atol = 1e-2)
    @test isapprox(xmp.nzval, x.nzval, atol = 1e-2)
end

@testset "Orthogonal Matching Pursuit" begin
    n, m, k = 32, 64, 4
    A, x, b = sparse_data(n = n, m = m, k = k)
    xomp = omp(A, b, k)

    # noiseless
    @test xomp.nzind == x.nzind
    @test xomp.nzval ≈ x.nzval

    σ = 1e-2 # slightly noisy
    @. b += σ*randn()
    xomp = omp(A, b, k)
    # noiseless
    @test xomp.nzind == x.nzind
    @test isapprox(xomp.nzval, x.nzval, atol = 5σ)
end

@testset "Optimized OMP" begin
    n, m, k = 32, 64, 4
    A, x, b = sparse_data(n = n, m = m, k = k)
    xomp = oomp(A, b, k)

    # noiseless
    @test xomp.nzind == x.nzind
    @test xomp.nzval ≈ x.nzval

    σ = 1e-2 # slightly noisy
    @. b += σ*randn()
    xomp = omp(A, b, k)
    # noiseless
    @test xomp.nzind == x.nzind
    @test isapprox(xomp.nzval, x.nzval, atol = 5σ)
end

@testset "OMP with replacement" begin
    n, m, k = 32, 64, 3
    A, x, b = sparse_data(n = n, m = m, k = k)

    δ = 1e-6
    # xomp = zero(x)
    # @. xomp[1:k] = 1
    xomp = ompr(A, b, k, δ)

    # noiseless
    @test xomp.nzind == x.nzind
    @test xomp.nzval ≈ x.nzval

    # σ = 1e-2 # slightly noisy
    # @. b += σ*randn()
    # xomp = ompr(A, b, k)
    # # noiseless
    # @test xomp.nzind == x.nzind
    # @test isapprox(xomp.nzval, x.nzval, atol = 5σ)
end

@testset "Subspace Pursuit" begin
    n, m, k = 32, 64, 3
    A, x, b = sparse_data(n = n, m = m, k = k)

    xssp = sp(A, b, k)
    # noiseless
    @test xssp.nzind == x.nzind
    @test isapprox(xssp.nzval, x.nzval)

    σ = 1e-2 # noisy
    @. b += σ*randn()
    xssp = sp(A, b, k, 1e-2)
    # noiseless
    @test xssp.nzind == x.nzind
    @test isapprox(xssp.nzval, x.nzval, atol = 5σ)
end

@testset "Relevance Matching Pursuit" begin
    n, m, k = 32, 64, 4
    A, x, b = sparse_data(n = n, m = m, k = k)
    δ = 1e-6
    xrmp = rmp(A, b, δ, rescale = true)
    # noiseless
    @test xrmp.nzind == x.nzind
    @test xrmp.nzval ≈ x.nzval

    σ = 1e-3 # slightly noisy
    @. b += σ*randn()
    xrmp = rmp(A, b, 1e-2, rescale = false)
    # noiseless

    @test xrmp.nzind == x.nzind
    @test isapprox(xrmp.nzval, x.nzval, atol = 1e-2)
end

end # TestMatchingPursuit
