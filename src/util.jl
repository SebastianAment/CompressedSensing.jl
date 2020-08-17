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
		μ = mean(x, dims = 1)
        @. y = x - (1-ε) * μ
    end
    p!(x) = p!(x, x)
    return p!
end
# precondition!(A::AbstractVecOrMat, ε::Real) = preconditioner(ε)(A)

# related but not identical to the one used in:
# "Preconditioned Multiple Orthogonal Least Squares and Applications in Ghost Imaging via Sparsity Constraint"
# preconditions using U*S⁻¹*U' where U, S, V is svd of A
function preconditioner(A::AbstractMatrix{Float64}, min_σ::Real = 1e-6)
    svdA = svd(A)
    U, S, V = svdA.U, svdA.S, svdA.V
	k = size(U, 2)
	function p!(y, x, z)
		size(y) == size(x) || throw(DimensionMismatch("size(x) ≠ size(y)"))
		mul!(z, U', x)
		z ./= max.(S, min_σ)
		mul!(y, U, z)
	end
	p!(y::AbstractMatrix, x::AbstractMatrix) = p!(y, x, similar(x, k, size(x, 2)))
	p!(y::AbstractVector, x::AbstractVector) = p!(y, x, similar(x, k))
	p!(x::AbstractVecOrMat) = p!(x, x)
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

########################### SparseArrays conveniences ##########################
# drops the ith index-value pair of x, if it is in nzind
function dropindex!(x::SparseVector, i::Int)
	j = findfirst(==(i), x.nzind)
	if !isnothing(j)
		deleteat!(x.nzind, j)
		deleteat!(x.nzval, j)
	end
    return x
end

function Base.findmin(f, x::AbstractVector)
    k, m = 0, Inf
    for (i, xi) in enumerate(x)
        fxi = f(xi)
        if fxi < m
            k, m = i, fxi
        end
    end
    return m, k
end
Base.argmin(f, x::AbstractVector) = findmin(f, x)[2]
function Base.findmax(f, x::AbstractVector)
	g(x) = -f(x)
	m, k = findmin(g, x)
	return -m, k
end
Base.argmax(f, x::AbstractVector) = findmax(f, x)[2]

# # TODO: this should be in util somewhere
# using StatsBase: mean, std
# # makes x mean zero along dimensions dims
# function center!(x::AbstractVecOrMat, ε = 1e-6; dims = :)
#     μ, σ = mean(x, dims = dims), std(x, dims = dims)
#     @. x = (x - μ) / σ
#     # @. x = (x - (1-ε)*μ) / σ
#     return x, μ, σ
# end
# Base.inv(::typeof(center!)) = uncenter
# function uncenter(x::AbstractVecOrMat, μ, σ)
#     @. x = x * σ + μ
#     x, y
# end