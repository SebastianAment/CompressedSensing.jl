module TestForward
using CompressedSensing
using CompressedSensing: fr, sparse_data

using Test
using LinearAlgebra
@testset "forward regression" begin
    n, k = 32, 4
    δ = .001
    A, x, b = sparse_data(n = n, m = n, k = k, min_x = √2δ) # needs to be determined
    ε = randn(n)
    ε .*= δ/norm(ε)
    y = b + ε
    xfr = fr(A, y, δ)
    # display(xfr)
    # display(x)
    @test x.nzind == xfr.nzind # support recovery
end


end # TestForward
