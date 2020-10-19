module TestTwoStage
using Test
using CompressedSensing: gaussian_data, perturb, srr, sp, ompr, sparse_vector

@testset "Two-Stage Algorithms" begin
    n, m, k = 32, 64, 3
    A, x, b = gaussian_data(n, m, k)
    δ = 1e-2
    y = perturb(b, δ/2)

    @testset "Stagewise Regression with Replacement" begin
        # noiseless
        xsrr = srr(A, b, k)
        @test xsrr.nzind == x.nzind
        @test xsrr.nzval ≈ x.nzval

        # noisy
        xsrr = srr(A, y, k)
        @test xsrr.nzind == x.nzind
        @test isapprox(xsrr.nzval, x.nzval, atol = 3δ)

        # special case
        x1 = sparse_vector(m, 1)
        xsrr = srr(A, A*x1, 1)
        @test xsrr.nzind == x1.nzind
        @test xsrr.nzval ≈ x1.nzval

        # l-step version of srr
        l = k
        # noiseless
        xsrr = srr(A, b, k, l = k)
        @test xsrr.nzind == x.nzind
        @test xsrr.nzval ≈ x.nzval

        # noisy
        xsrr = srr(A, y, k, l = k)
        @test xsrr.nzind == x.nzind
        @test isapprox(xsrr.nzval, x.nzval, atol = 3δ)

    end

    @testset "Subspace Pursuit" begin
        xsp = sp(A, b, k)
        # noiseless
        @test xsp.nzind == x.nzind
        @test xsp.nzval ≈ x.nzval

        # noisy
        xssp = sp(A, y, k, δ)
        @test xssp.nzind == x.nzind
        @test isapprox(xssp.nzval, x.nzval, atol = 3δ)
    end

    @testset "OMP with replacement" begin
        # noiseless
        xompr = ompr(A, b, k, 1e-6)
        @test xompr.nzind == x.nzind
        @test xompr.nzval ≈ x.nzval

        # slightly noisy
        xomp = ompr(A, y, k, δ)
        @test xomp.nzind == x.nzind
        @test isapprox(xomp.nzval, x.nzval, atol = 3δ)
    end
end
end # TestTwoStage
