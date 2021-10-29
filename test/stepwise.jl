module TestStepwise
using Test
using CompressedSensing: gaussian_data, perturb, rmp, foba

@testset "Stepwise Regression Algorithms" begin
    n, m, k = 32, 64, 3
    A, x, b = gaussian_data(n, m, k)
    δ = 1e-2
    y = perturb(b, δ)

    @testset "Relevance Matching Pursuit" begin
        xrmp = rmp(A, b, k)
        # noiseless
        @test xrmp.nzind == x.nzind
        @test xrmp.nzval ≈ x.nzval

        # noisy
        xrmp = rmp(A, y, δ)
        @test xrmp.nzind == x.nzind
        @test isapprox(xrmp.nzval, x.nzval, atol = 2δ)

        # multiple outer loop iterations
        xrmp = rmp(A, y, δ, 3)
        @test xrmp.nzind == x.nzind
        @test isapprox(xrmp.nzval, x.nzval, atol = 2δ)
    end

    @testset "FoBa" begin
        # noiseless
        xfoba = foba(A, b, δ)
        @test xfoba.nzind == x.nzind
        @test xfoba.nzval ≈ x.nzval

        # noisy
        xfoba = foba(A, y, δ)
        @test xfoba.nzind == x.nzind
        @test isapprox(xfoba.nzval, x.nzval, atol = 2δ)
    end
end

end
