function sparse_data(;n = 32, m = 64, k = 3, min_x = 0., rescaled = true)
    A = randn(n, m)
    if rescaled
        A .-= mean(A, dims = 1)
        A ./= sqrt.(sum(abs2, A, dims = 1))
    end
    x = spzeros(m)
    ind = sort!(sample(1:m, k, replace = false))
    @. x[ind] = $rand((-1,1)) * max(abs(randn()), min_x)

    b = A*x
    A, x, b
end

# calculates mutual coherence
function coherence(A::AbstractMatrix)
    μ = zero(eltype(A))
    for (i, ai) in enumerate(eachcol(A))
        for aj in eachcol(view(A, :, 1:i-1))
            μ = max(μ, abs(dot(ai, aj)))
        end
    end
    return μ
end

# Babel function, see GREED IS GOOD: ALGORITHMIC RESULTS FOR SPARSE APPROXIMATION
function babel(A::AbstractMatrix, k::Integer)
    μ₁ = zero(eltype(A))
    inner = similar(@view(A[1, :]))
    for (i, ai) in enumerate(eachcol(A))
        mul!(inner, A', ai)
        @. inner = abs(inner)
        inner[i] = 0 # inner product with self does not count
        partialsort!(inner, 1:k, rev = true) # pick largest k inner products
        μ₁ = max(μ₁, sum(@view inner[1:k]))
    end
    return μ₁
end
