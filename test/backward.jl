module TestBackward
using CompressedSensing
using CompressedSensing: lace, br, sparse_data

using Test
using LinearAlgebra
@testset "backward regression" begin
    n, k = 32, 3
    δ = .01
    A, x, b = sparse_data(n = n, m = n, k = k, min_x = √2δ) # needs to be determined
    ε = randn(n)
    ε .*= δ/norm(ε)
    y = b + ε
    xbr = br(A, y, δ)
    # display(xbr)
    # display(x)
    @test x.nzind == xbr.nzind # support recovery
end

function residual_magnitude(n::Int)
    A = randn(n, n-1)
    a = randn(n)
    norm(a-A\a)
end

end # TestBackward
