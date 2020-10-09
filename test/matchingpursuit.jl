module TestMatchingPursuit
using Test
using LinearAlgebra
using CompressedSensing: MP, mp, OMP, omp, SP, sp, rmp, sparse_data,
                        ols, ompr, lmp, foba
using SparseArrays
# TODO: pool tests of similar algorithms, e.g.: omp, ols, ompr
@testset "Matching Pursuits" begin
    n, m, k = 32, 64, 3
    A, x, b = sparse_data(n = n, m = m, k = k)
    σ = 1e-2 # slightly noisy
    ε = randn(n)
    ε .*= σ/2norm(ε)

    @testset "Matching Pursuit" begin
        xmp = mp(A, b, 3k) # giving more iterations to optimize
        @test isapprox(A*xmp, b, atol = 2σ)
        @test isapprox(xmp.nzval, x.nzval, atol = 2σ)
    end

    @testset "Orthogonal Matching Pursuit" begin
        # noiseless
        xomp = omp(A, b, k)
        @test xomp.nzind == x.nzind
        @test xomp.nzval ≈ x.nzval
        # noisy
        xomp = omp(A, b + ε, k)
        @test xomp.nzind == x.nzind
        @test isapprox(xomp.nzval, x.nzval, atol = 2σ)
    end

    @testset "OMP with replacement" begin
        # noiseless
        δ = 1e-6
        xompr = ompr(A, b, k, δ)
        @test xompr.nzind == x.nzind
        @test xompr.nzval ≈ x.nzval
        # slightly noisy
        xomp = ompr(A, b + ε, k)
        @test xomp.nzind == x.nzind
        @test isapprox(xomp.nzval, x.nzval, atol = 5σ)
    end

    @testset "Subspace Pursuit" begin
        xssp = sp(A, b, k)
        # noiseless
        @test xssp.nzind == x.nzind
        @test isapprox(xssp.nzval, x.nzval)
        # noiseless
        xssp = sp(A, b + ε, k, 1e-2)
        @test xssp.nzind == x.nzind
        @test isapprox(xssp.nzval, x.nzval, atol = 5σ)
    end

    @testset "Relevance Matching Pursuit" begin
        δ = 1e-6
        xrmp = rmp(A, b, k)
        # noiseless
        @test xrmp.nzind == x.nzind
        @test xrmp.nzval ≈ x.nzval

        xlmp = lmp(A, b, k)
        # noiseless
        @test xlmp.nzind == x.nzind
        @test xlmp.nzval ≈ x.nzval

        # noisy
        xrmp = rmp(A, b + ε, 1e-2)
        @test xrmp.nzind == x.nzind
        @test isapprox(xrmp.nzval, x.nzval, atol = 5σ)
    end

    @testset "FoBa" begin
        δ = 1e-6
        xfoba = foba(A, b, δ)
        # noiseless
        @test xfoba.nzind == x.nzind
        @test xfoba.nzval ≈ x.nzval
    end

end

end # TestMatchingPursuit
