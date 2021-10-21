abstract type AbstractMatchingPursuit{T} <: Update{T} end

# TODO: lazy evaluation for OMP via approximate submodularity? early dropping?
# TODO: rename Ar for δ to be consistent with generalized matching pursuits

############################# Matching Pursuit #################################
# should MP, OMP, have a k parameter?
# SP needs one, but for MP could be handled in outer loop
# WARNING: requires A to have unit norm columns
struct MatchingPursuit{T, AT<:AbstractMatOrFac{T}, B<:AbstractVector{T}} <: AbstractMatchingPursuit{T}
    A::AT
    b::B
    k::Int # maximum number of non-zeros

    # temporary storage
    r::B # residual
    Ar::B # inner products between measurement matrix and residual
    # a approximation # can be updated efficiently
end
const MP = MatchingPursuit
function MP(A::AbstractMatOrFac, b::AbstractVector, k::Integer)
    n, m = size(A)
    T = eltype(A)
    r, Ar = zeros(T, n), zeros(T, m)
    MP(A, b, k, r, Ar)
end

function update!(P::MP, x::AbstractVector = spzeros(size(P.A, 2)))
    nnz(x) ≥ P.k && return x # return if the maximum number of non-zeros was reached
    residual!(P, x) # TODO: could just update approximation directly for mp
    i = argmaxinner!(P)
    x[i] += dot(@view(P.A[:, i]), P.r) # add non-zero index to x
    return x
end

# calculates k-sparse approximation to Ax = b via matching pursuit
function mp(A::AbstractMatOrFac, b::AbstractVector, k::Int, x = spzeros(size(A, 2)))
    P = MP(A, b, k)
    for i in 1:k
        update!(P, x)
    end
    x
end

###################### Orthogonal Matching Pursuit #############################
# could extend: preconditioning, non-negativity constraint
struct OrthogonalMatchingPursuit{T, AT<:AbstractMatOrFac{T}, B<:AbstractVector{T},
                            V<:AbstractVector{T}, FT} <: AbstractMatchingPursuit{T}
    A::AT
    b::B
    k::Int # maximum number of non-zeros

    # temporary storage
    r::V # residual
    Ar::V # inner products between measurement matrix and residual
    AiQR::FT # updatable QR factorization
end
const OMP = OrthogonalMatchingPursuit
function OMP(A::AbstractMatOrFac, b::AbstractVector, k::Integer = size(A, 1))
    n, m = size(A)
    T = eltype(A)
    r, Ar = zeros(T, n), zeros(T, m)
    AiQR = UpdatableQR(T, n, n) # initializing empty qr factorization with maximum rank n
    OMP(A, b, k, r, Ar, AiQR)
end

function update!(P::OMP, x::AbstractVector = spzeros(size(P.A, 2)))
    nnz(x) < size(P.A, 1) || return x
    residual!(P, x)
    i = argmaxinner!(P)
    ∉(i, x.nzind) || return x
    addindex!(x, P, i)
    ldiv!!(x.nzval, P.AiQR, P.b, P.r)
    return x
end

# approximately solves Ax = b with error tolerance ε it at most k steps
function omp(A::AbstractMatOrFac, b::AbstractVector, ε::Real, k::Int = size(A, 1))
    ε ≥ 0 || throw("ε = $ε has to be non-negative")
    P = OMP(A, b)
    x = spzeros(size(A, 2))
    for i in 1:k
        update!(P, x)
        norm(residual!(P, x)) ≥ ε || break
    end
    return x
end
# calculates k-sparse approximation to Ax = b via orthogonal matching pursuit
function omp(A::AbstractMatOrFac, b::AbstractVector, k::Int)
    omp(A, b, eps(eltype(A)), k)
end

function omp(A::AbstractMatOrFac, b::AbstractVector;
    max_residual = eps(eltype(A)), sparsity = size(A, 2))
    omp(A, b, max_residual, sparsity)
end

####################### Matching Pursuit Helpers ###############################
# calculates residual of
function residual!(P::AbstractMatchingPursuit, x::AbstractVector)
    residual!(P.r, P.A, x, P.b)
end
function residual!(P, x::AbstractVector)
    residual!(P.r, P.A, x, P.b)
end
function residual!(r::AbstractVector, A::AbstractMatOrFac, x::AbstractVector, b::AbstractVector)
    copyto!(r, b)
    mul!(r, A, x, -1, 1)
end

# adds non-zero at i, adds ith column of A to AiQR, and solves resulting ls-problem
function addindex!(x::SparseVector, P::AbstractMatchingPursuit, i::Int)
    addindex!(x, P.AiQR, @view(P.A[:, i]), i)
    return x, P
end

# overwrites r with intermediate result and y with final result, leaves b intact
function ldiv!!(y::AbstractVector, AiQR::UpdatableQR, b::AbstractVector, r::AbstractVector)
    @. r = b # reuse P.r as temporary storage to execute the least-squares solve with P.AiQR
    y .= ldiv!(AiQR, r)
end

########################### acquisition criteria ###############################
# WARNING: need to compute residual first
# returns index of atom with largest dot product with residual
@inline function argmaxinner!(P::AbstractMatchingPursuit) # 0 alloc
    mul!(P.Ar, P.A', P.r)
    @. P.Ar = abs(P.Ar)
    argmax(P.Ar)
end

# returns indices of k atoms with largest inner products with residual
@inline function argmaxinner!(P::AbstractMatchingPursuit, k::Int)
    mul!(P.Ar, P.A', P.r)
    @. P.Ar = abs(P.Ar)
    partialsortperm(P.Ar, 1:k, rev = true)
end

function random_acquisition!(P::AbstractMatchingPursuit, x::SparseVector, k::Int)
    ind = sample(1:length(x), k, replace = false) # or could initialize with max inner products
    sort!(ind)
    @. x[ind] = NaN
    for i in ind
        add_column!(P.AiQR, @view(P.A[:, i]))
    end
    ldiv!(x.nzval, P.AiQR, P.b)
    return x
end

# using oblivious algorithm to initialize P
function oblivious_acquisition!(P::AbstractMatchingPursuit, x::SparseVector, k::Int = P.k)
    residual!(P, x)
    ind = argmaxinner!(P, k)
    sort!(ind)
    @. x[ind] = NaN
    for i in ind
        add_column!(P.AiQR, @view P.A[:, i])
    end
    ldiv!(x.nzval, P.AiQR, P.b)
end

################################################################################
factorize!(P::AbstractMatchingPursuit, x::SparseVector) = factorize!(P, x.nzind)
# IDEA: could have non-qr-based solution
@inline function factorize!(P::AbstractMatchingPursuit, nzind::Vector{<:Int})
    n = size(P.A, 1)
    k = length(nzind)
    Ai = reshape(@view(P.Ai[1:n*k]), n, k)
    @. Ai = @view(P.A[:, nzind])
    return qr!(Ai) # this still allocates a little memory
end
