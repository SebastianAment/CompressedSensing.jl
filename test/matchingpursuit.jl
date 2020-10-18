module TestMatchingPursuit
using Test
using LinearAlgebra
using CompressedSensing: MP, mp, OMP, omp, SP, sp, rmp, sparse_data,
                        ols, ompr, srr, foba, perturb
using SparseArrays
# algorithms = [(A, b) -> mp(A, b, 3k),
#               (A, b) -> omp(A, b, k),
#               (A, b) -> ompr(A, b, k)]
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

end

end # TestMatchingPursuit
