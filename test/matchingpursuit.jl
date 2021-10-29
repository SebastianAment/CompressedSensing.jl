module TestMatchingPursuit
using Test
using LinearAlgebra
using CompressedSensing: MP, mp, OMP, omp, SP, sp, rmp, sparse_data,
                        ols, ompr, srr, foba, perturb, gomp
using SparseArrays
# NOTE: these tests may rarely fail, because the randomly generated data is "hard"
# a small failure probability is theoretically expected
@testset "Matching Pursuits" begin
    n, m, k = 32, 64, 3
    A, x, b = sparse_data(n = n, m = m, k = k)
    δ = 1e-2 # slightly noisy
    y = perturb(b, δ/2)

    @testset "Matching Pursuit" begin
        xmp = mp(A, b, 10k) # giving more iterations to optimize
        @test isapprox(A*xmp, b, atol = 3δ)
        @test isapprox(xmp, x, atol = 3δ)
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

    @testset "Generalized Orthogonal Matching Pursuit" begin
        n, m, k = 64, 128, 7
        A, x, b = sparse_data(n = n, m = m, k = k)
        y = perturb(b, δ/2)
        l = 3 # identify three atoms at a time
        # noiseless
        xgomp = gomp(A, b, l, k)
        @test xgomp.nzind == x.nzind
        @test xgomp.nzval ≈ x.nzval
        # noisy
        xgomp = gomp(A, y, l, k)
        @test xgomp.nzind == x.nzind
        @test isapprox(xgomp.nzval, x.nzval, atol = 2δ)
    end
end

end # TestMatchingPursuit
