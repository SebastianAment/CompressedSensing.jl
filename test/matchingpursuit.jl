module TestMatchingPursuit
using Test
using LinearAlgebra
using CompressedSensing: MP, mp, OMP, omp, SP, sp, rmp, sparse_data,
                        ols, ompr, lmp, foba, perturb
using SparseArrays
# TODO: pool tests of similar algorithms, e.g.: omp, ols, ompr
@testset "Matching Pursuits" begin
    n, m, k = 32, 64, 3
    A, x, b = sparse_data(n = n, m = m, k = k)
    δ = 1e-2 # slightly noisy
    y = perturb(b, δ/2)

    @testset "Matching Pursuit" begin
        xmp = mp(A, b, 3k) # giving more iterations to optimize
        @test isapprox(A*xmp, b, atol = 2δ)
        @test isapprox(xmp.nzval, x.nzval, atol = 2δ)
    end

    @testset "Orthogonal Matching Pursuit" begin
        # noiseless
        xomp = omp(A, b, k)
        @test xomp.nzind == x.nzind
        @test xomp.nzval ≈ x.nzval
        # noisy
        xomp = omp(A, y, k)
        @test xomp.nzind == x.nzind
        @test isapprox(xomp.nzval, x.nzval, atol = 2δ)
    end

    @testset "OMP with replacement" begin
        # noiseless
        # δ = 1e-6
        xompr = ompr(A, b, k, 1e-6)
        @test xompr.nzind == x.nzind
        @test xompr.nzval ≈ x.nzval

        # slightly noisy
        xomp = ompr(A, y, k)
        @test xomp.nzind == x.nzind
        @test isapprox(xomp.nzval, x.nzval, atol = 5δ)
    end

    @testset "Subspace Pursuit" begin
        xssp = sp(A, b, k)
        # noiseless
        @test xssp.nzind == x.nzind
        @test isapprox(xssp.nzval, x.nzval)
        # noiseless
        xssp = sp(A, y, k, δ)
        @test xssp.nzind == x.nzind
        @test isapprox(xssp.nzval, x.nzval, atol = 5δ)
    end

    @testset "Relevance Matching Pursuit" begin
        xrmp = rmp(A, b, k)
        # noiseless
        @test xrmp.nzind == x.nzind
        @test xrmp.nzval ≈ x.nzval

        xlmp = lmp(A, b, k)
        # noiseless
        @test xlmp.nzind == x.nzind
        @test xlmp.nzval ≈ x.nzval

        # noisy
        xrmp = rmp(A, y, δ)
        @test xrmp.nzind == x.nzind
        @test isapprox(xrmp.nzval, x.nzval, atol = 5δ)

        # multiple outer loop iterations
        xrmp = rmp(A, y, δ, 3)
        @test xrmp.nzind == x.nzind
        @test isapprox(xrmp.nzval, x.nzval, atol = 5δ)

    end

    @testset "FoBa" begin
        xfoba = foba(A, b, δ)
        # noiseless
        @test xfoba.nzind == x.nzind
        @test xfoba.nzval ≈ x.nzval
    end

end

end # TestMatchingPursuit
