module TestUtil
using Test
using LinearAlgebra
using CompressedSensing: coherence, babel, sparse_data

@testset "dictionary analysis" begin
    n, m, k = 64, 128, 3
    A, x, b = sparse_data(n = n, m = m, k = k, rescaled = true)
    μ = coherence(A)
    @test μ isa Real
    @test 0 < μ
    @test babel(A, 1) ≈ μ
    @test babel(A, 2) < 2coherence(A)
end

end
