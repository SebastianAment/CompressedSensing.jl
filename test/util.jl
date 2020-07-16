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
    @. A = abs(A) # preconditioning important for coherence dictionaries
    normalize!(A)
    @. x = abs(x)
    b = A*x

    # svd-based preconditioner, generally yields good babel function improvements
    PA = copy(A)
    PA = normalize!(preconditioner(A)(PA)) # renormalize after preconditioning
    μ = cumbabel(A, k)
    Pμ = cumbabel(PA, k)
    for i in 1:k  # test that preconditioner decreases babel function
        @test Pμ[i] < μ[i]
    end
    # mean preconditioner, babel improvement of MA good but not as large as PA
    ε = 1e-6
    MA = copy(A)
    normalize!(preconditioner(ε)(MA))
    Mμ = cumbabel(MA, k)
    for i in 1:k  # test that preconditioner decreases babel function
        @test Mμ[i] < μ[i]
    end
end

end

# display(μ)
# display(Mμ)
# display(Pμ)

# PPA = copy(PA)
# repeated application does change babel function, but not meaningfully
# for _ in 1:6
#     PPA = normalize!(preconditioner(PPA)(copy(PPA)))
#     PPμ = cumbabel(PPA, k)
#     display(PPμ)
# end
