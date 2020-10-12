# helper
colnorms(A::AbstractMatrix) = [norm(a) for a in eachcol(A)]

function samesupport(x::AbstractVector, y::AbstractVector)
    samesupport(sparse(x), sparse(y))
end
function samesupport(x::SparseVector, y::SparseVector)
    sort!(x.nzind) == sort!(y.nzind)
end

########################## synthetic data generators ###########################
# creates random k-sparse vector with ±1 as entries, or gaussian depending on flag
function sparse_vector(m::Int, k::Int, gaussian::Bool = false)
    m ≥ k || throw("m = $m < $k = k")
    x = spzeros(m)
    ind = sort!(sample(1:m, k, replace = false))
    x[ind] .= gaussian ? randn(k) : rand((-1, 1), k)
    return x
end

function sparse_data(;n = 32, m = 64, k = 3, rescaled = true)
    A = randn(n, m)
    if rescaled
        ε = 1e-6
        A .-= ε*mean(A, dims = 1)
        A ./= sqrt.(sum(abs2, A, dims = 1))
    end
	x = sparse_vector(m, k)
    b = A*x
    return A, x, b
end
const gaussian_data = sparse_data
sparse_data(n, m, k) = sparse_data(n = n, m = m, k = k)
function correlated_data(n, m, k; normalized = true)
    U = randn(n, n)
    V = randn(n, m)
    S = Diagonal([1/i^2 for i in 1:n])
    A = U*S*V
    # normalize
    if normalized
        A ./= sqrt.(sum(abs2, A, dims = 1))
    end
    x = sparse_vector(m, k)
    b = A*x
    A, x, b
end
const coherent_data = correlated_data

# random perturbation of b with norm δ
function perturb!(b::AbstractVector, δ::Real)
    e = randn(size(b))
    e *= δ/norm(e)
    b .+= e # perturb
end
perturb(b, δ) = perturb!(copy(b), δ)


########################### dictionary preconditioners #########################
function normalize!(A::AbstractVecOrMat)
    A ./= colnorms(A)'
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
function addindex!(x::SparseVector, AiQR::Union{<:UQR, <:PUQR}, a::AbstractVector, i::Int)
    x[i] = NaN   # add non-zero index to active set
    # efficiently update qr factorization using Givens rotations
    qr_i = findfirst(==(i), x.nzind) # index in qr where new atom should be added
    add_column!(AiQR, a, qr_i)
    return x, AiQR
end
const addind! = addindex!

# drops the ith index-value pair of x, if it is in nzind
function dropindex!(x::SparseVector, i::Int)
	j = findfirst(==(i), x.nzind)
	if !isnothing(j)
		deleteat!(x.nzind, j)
		deleteat!(x.nzval, j)
	end
    return x
end
const dropind! = dropindex!
dropind!(x::SparseVector, i::AbstractVector{<:Int}) = dropind!.((x,), i)
# simultaneously deletes an index of x and removes a column of a UpdatableQR
function dropindex!(x::SparseVector, AiQR::Union{<:UQR, <:PUQR}, i::Int)
    j = findfirst(==(i), x.nzind)
    if !isnothing(j)
		_dropindex!(x, AiQR, j)
    end
    return x, AiQR
end
# WARNING: drops index j into x,nzval, NOT into x, as dropindex!
function _dropindex!(x::SparseVector, AiQR::Union{<:UQR, <:PUQR}, j::Int)
	deleteat!(x.nzind, j)
	deleteat!(x.nzval, j)
	remove_column!(AiQR, j)
	return x, AiQR
end

function SparseArrays.droptol!(x::SparseVector, AiQR::Union{<:UQR, <:PUQR}, tol::Real)
    for i in reverse(eachindex(x.nzind)) # reverse is necessary to not mess up indexing into QR factorization
        if abs(x.nzval[i]) ≤ tol
            remove_column!(P.AiQR, i)
        end
    end
    return droptol!(x, tol), AiQR
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
