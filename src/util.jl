function sparse_data(;n = 32, m = 64, k = 3, min_x = 0., rescaled = true)
    A = randn(n, m)
    if rescaled
        ε = 1e-6
        A .-= ε*mean(A, dims = 1)
        A ./= sqrt.(sum(abs2, A, dims = 1))
    end
    x = spzeros(m)
    ind = sort!(sample(1:m, k, replace = false))
    @. x[ind] = $rand((-1,1)) * max(abs(randn()), min_x)

    b = A*x
    A, x, b
end

########################### dictionary preconditioners #########################
function normalize!(A::AbstractVecOrMat)
    A ./= sqrt.(sum(abs2, A, dims = 1))
end

# see "On the Uniqueness of Nonnegative Sparse Solutions to Underdetermined Systems of Equations", Bruckstein 2008
function preconditioner(ε::Real)
    function p!(y, x)
        y .= x .- (1-ε)*mean(x, dims = 1)
    end
    p!(x) = p!(x, x)
    return p!
end
# precondition!(A::AbstractVecOrMat, ε::Real) = preconditioner(ε)(A)

# related but not identical to the one used in:
# "Preconditioned Multiple Orthogonal Least Squares and Applications in Ghost Imaging via Sparsity Constraint"
function preconditioner(A::AbstractMatrix)
    svdA = svd(A)
    U, S = svdA.U, svdA.S
    function p!(y, x)
        z = similar(x)
        mul!(z, U', x)
        z ./= S
        mul!(y, U, z)
    end
    p!(x) = p!(x, x)
    return p!
end
function preconditioner(A::AbstractMatrix{<:Real}, min_σ::Real = 1e-6)
    svdA = svd(A)
    U, S, V = svdA.U, svdA.S, svdA.V
    function p!(y, x)
        size(y) == size(x) || throw(DimensionMismatch("size(x) ≠ size(y)"))
        z = similar(x, (size(U, 2), size(x, 2)))
        mul!(z, U', x)
        z ./= max.(S, min_σ)
        mul!(y, U, z)
    end
    p!(x) = p!(x, x)
    return p!
end
precondition!(A::AbstractMatrix) = preconditioner(A)(A)

############################# dictionary analysis ##############################
# calculates mutual coherence
coherence(A::AbstractMatrix) = babel(A, 1)

# Babel function, see GREED IS GOOD: ALGORITHMIC RESULTS FOR SPARSE APPROXIMATION
babel(A::AbstractMatrix, k::Integer) = cumbabel(A, k)[k]

Base.cumsum!(x::AbstractArray) = cumsum!(x, x)
# calculates all babel function values from 1 to k
function cumbabel(A::AbstractMatrix, k::Integer)
    μ₁ = zeros(eltype(A), k)
    inner = similar(A, size(A, 2))
    for (i, ai) in enumerate(eachcol(A))
        mul!(inner, A', ai)
        @. inner = abs(inner)
        inner[i] = 0 # inner product with self does not count
        partialsort!(inner, 1:k, rev = true) # pick largest k inner products
        innerk = @view inner[1:k]
        μ₁ .= max.(μ₁, cumsum!(innerk))
    end
    return μ₁
end
