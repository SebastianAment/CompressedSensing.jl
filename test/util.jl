module TestUtil
using Test
using LinearAlgebra
using CompressedSensing: cumbabel, coherence, babel, sparse_data, precondition!, normalize!, preconditioner

@testset "dictionary analysis" begin
    n, m, k = 64, 128, 16
    A, x, b = sparse_data(n = n, m = m, k = k, rescaled = true)
    μ = coherence(A)
    @test μ isa Real
    @test 0 < μ
    @test babel(A, 1) ≈ μ
    μ₁ = cumbabel(A, k)
    @test μ₁ ≈ babel.((A,), 1:k)
    tol = 1e-12 # tolerance for violation of inequality
    for (i, μ_i) in enumerate(μ₁)
        @test μ_i ≤ i*μ + tol
    end
end

@testset "preconditioning" begin
    n, m, k = 64, 128, 6
    A, x, b = sparse_data(n = n, m = m, k = k, rescaled = false)
    @. A = abs(A)
    normalize!(A)
    @. x = abs(x)
    b = A*x
    P! = preconditioner(A)
    PA = P!(copy(A))
    for i in 1:k # test that preconditioner decreases babel function
        @test babel(PA, k) < babel(A, k)
    end
end

end
