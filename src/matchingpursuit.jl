abstract type AbstractMatchingPursuit{T} <: Update{T} end

# TODO: lazy evaluation for OMP via approximate submodularity? early dropping?
# TODO: rename Ar for δ to be consistent with generalized matching pursuits

############################# Matching Pursuit #################################
# should MP, OMP, have a k parameter?
# SP needs one, but for MP could be handled in outer loop
# WARNING: requires A to have unit norm columns
struct MatchingPursuit{T, AT<:AbstractMatrix{T}, B<:AbstractVector{T}} <: AbstractMatchingPursuit{T}
    A::AT
    b::B
    k::Int # maximum number of non-zeros

    # temporary storage
    r::B # residual
    Ar::B # inner products between measurement matrix and residual
    # a approximation # can be updated efficiently
end
const MP = MatchingPursuit
function MP(A::AbstractMatrix, b::AbstractVector, k::Integer)
    n, m = size(A)
    T = eltype(A)
    r, Ar = zeros(T, n), zeros(T, m)
    MP(A, b, k, r, Ar)
end

function update!(P::MP, x::AbstractVector = spzeros(size(P.A, 2)))
    nnz(x) ≥ P.k && return x # return if the maximum number of non-zeros was reached
    residual!(P, x) # TODO: could just update approximation directly for mp
    i = argmaxinner!(P)
    x[i] += dot(@view(P.A[:,i]), P.r) # add non-zero index to x
    return x
end

# calculates k-sparse approximation to Ax = b via matching pursuit
function mp(A::AbstractMatrix, b::AbstractVector, k::Int, x = spzeros(size(A, 2)))
    P = MP(A, b, k)
    for i in 1:k
        update!(P, x)
    end
    x
end

###################### Orthogonal Matching Pursuit #############################
# could extend: preconditioning, non-negativity constraint
struct OrthogonalMatchingPursuit{T, AT<:AbstractMatrix{T}, B<:AbstractVector{T},
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
function OMP(A::AbstractMatrix, b::AbstractVector, k::Integer = size(A, 1))
    n, m = size(A)
    T = eltype(A)
    r, Ar = zeros(T, n), zeros(T, m)
    # AiQR = UpdatableQR(reshape(A[:, 1], :, 1))
    AiQR = PUQR(reshape(A[:, 1], :, 1)) # PermutedUpdatableQR
    remove_column!(AiQR) # start with empty factorization
    OMP(A, b, k, r, Ar, AiQR)
end

function update!(P::OMP, x::AbstractVector = spzeros(size(P.A, 2)))
    nnz(x) < size(P.A, 1) || return x
    residual!(P, x)
    i = argmaxinner!(P)
    ∉(i, x.nzind) || return x
    addindex!(x, P, i)
    ldiv!(x.nzval, P.AiQR, P.b)
    return x
end

# approximately solves Ax = b with error tolerance δ it at most k steps
function omp(A::AbstractMatrix, b::AbstractVector, ε::Real, k::Int = size(A, 1))
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
function omp(A::AbstractMatrix, b::AbstractVector, k::Int)
    omp(A, b, eps(eltype(A)), k)
end

function omp(A::AbstractMatrix, b::AbstractVector;
    max_residual = eps(eltype(A)), sparsity = size(A, 2))
    omp(A, b, max_residual, sparsity)
end

################################## Subspace Pursuit ############################
struct SubspacePursuit{T, AT<:AbstractMatrix{T}, B<:AbstractVector{T}} <: AbstractMatchingPursuit{T}
    A::AT
    b::B
    k::Int # maximum number of non-zeros

    # temporary storage
    r::B # residual
    Ar::B # inner products between measurement matrix and residual
    Ai::AT # space for A[:, x.nzind] and its qr factorization
end
const SP = SubspacePursuit

function SP(A::AbstractMatrix, b::AbstractVector, k::Integer)
    2k > length(b) && error("2k = $(2k) > $(length(b)) = length(b) is invalid for Subspace Pursuit")
    n, m = size(A)
    T = eltype(A)
    r, Ar = copy(b), zeros(T, m)
    Ai = zeros(T, (n, 2k))
    SP(A, b, k, r, Ar, Ai)
end

# returns indices of k atoms with largest inner products with residual
# could use threshold on P.Ar for adaptive stopping
sp_index!(P::SP, k::Int = P.k) = argmaxinner!(P, k)
function sp_acquisition!(P::SP, x, k::Int = P.k)
    residual!(P, x)
    i = sp_index!(P, k)
    @. x[i] = NaN
    solve!(P, x)
end
# TODO: could pre-allocate nz arrays to be of length 2K
# TODO: could add efficient qr updating
function update!(P::SP, x::AbstractVector = spzeros(size(P.A, 2)), ε::Real = 0.)
    nnz(x) == P.k || throw("nnz(x) = $(nnz(x)) ≠ $(P.k) = k")
    sp_acquisition!(P, x)
    i = partialsortperm(abs.(x.nzval), 1:nnz(x)-P.k) # find the smallest atoms
    sort!(i) # for deleteat! to work, indices need to be sorted
    deleteat!(x.nzind, i)
    deleteat!(x.nzval, i)
    solve!(P, x) # optimize all active atoms
end

# calculates k-sparse approximation to Ax = b via subspace pursuit
# could also stop if indices are same between iterations
function sp(A::AbstractMatrix, b::AbstractVector, k::Int, δ::Real = 1e-12; maxiter = 16)
    P = SP(A, b, k)
    x = spzeros(size(A, 2))
    sp_acquisition!(P, x, P.k)
    for i in 1:maxiter
        update!(P, x)
        if norm(residual!(P, x)) < δ # if we found a solution with the required sparsity we're done
            break
        end
    end
    return x
end

############################# OMP with replacement #############################
struct OMPR{T, AT<:AbstractMatrix{T}, B<:AbstractVector{T}, FT} <: AbstractMatchingPursuit{T}
    A::AT
    b::B
    k::Int
    # l # number of atoms to replace

    # temporary storage
    r::B # residual
    Ar::B # inner products between measurement matrix and residual
    QA::AT # (k, m)
    AiQR::FT # updatable QR factorization of Ai
end

# similar to RMP, or OMP?
function OMPR(A::AbstractMatrix, b::AbstractVector, k::Int)
    n, m = size(A)
    T = eltype(A)
    r, Ar = zeros(T, n), zeros(T, m)
    # AiQR = UpdatableQR(reshape(A[:, 1], :, 1))
    AiQR = PUQR(reshape(A[:, 1], :, 1))
    QA = zeros(T, (n, m))
    remove_column!(AiQR) # start with empty factorization
    OMPR(A, b, k, r, Ar, QA, AiQR)
end

# η is stepsize
function update!(P::OMPR, x::AbstractVector, η::Real = 1.)
    nnz(x) == P.k || throw("nnz(x) = $(nnz(x)) ≠ $(P.k) = k")
    residual!(P, x)
    copy!(P.Ar, x)
    mul!(P.Ar, P.A', P.r, η, 1)
    begin # argmax of abs(P.Ar) over i not in x.nzind     # argmax( [abs(ar) for ar in P.Ar]
        m = 0.
        i = 0
        for (j, Arj) in enumerate(P.Ar)
            if j ∉ x.nzind
                fArj = abs(Arj)
                if fArj > m
                    m = fArj
                    i = j
                end
            end
        end
    end

    if i == 0
        return x
    end

    # add non-zero index to active set
    x[i] = NaN
    qr_i = findfirst(==(i), x.nzind) # index in qr where new atom should be added

    # least-squares solve for active atoms (this would be two-stage algorithm)
    # ldiv!(x.nzval, P.AiQR, P.b)

    # gradient descent for active atoms
    @. x.nzval = P.Ar[x.nzind]

    j = argmin(abs, x.nzval) # delete index
    deleteat!(x.nzind, j) # delete the value from array
    deleteat!(x.nzval, j)

    if qr_i ≠ j # update qr factorization using Givens rotations
        a = @view P.A[:, i]
        add_column!(P.AiQR, a, qr_i)
        remove_column!(P.AiQR, j)
    end

    # least-squares solve for active atoms
    ldiv!(x.nzval, P.AiQR, P.b)
    return x
end

# k is desired sparsity level
# l is cardinality of maximum replacement per iteration, l = k corresponds to sp
function ompr(A::AbstractMatrix, b::AbstractVector, k::Int, δ::Real = 1e-12,
                                x = spzeros(size(A, 2)); maxiter = size(A, 1))
    P = OMPR(A, b, k)
    if nnz(x) < P.k # make sure support set is of size k
        @. x = 0
        dropzeros!(x)
        oblivious_acquisition!(P, x, P.k)
    end
    resnorm = norm(residual!(P, x))
    for i in 1:maxiter
        oldnorm = resnorm
        update!(P, x)
        resnorm = norm(residual!(P, x))
        if resnorm ≤ δ || oldnorm ≤ resnorm
            break
        end
    end
    return x
end

####################### Matching Pursuit Helpers ###############################
# calculates residual of
function residual!(P::AbstractMatchingPursuit, x::AbstractVector)
    residual!(P.r, P.A, x, P.b)
end
function residual!(P, x::AbstractVector)
    residual!(P.r, P.A, x, P.b)
end
function residual!(r::AbstractVector, A::AbstractMatrix, x::AbstractVector, b::AbstractVector)
    copyto!(r, b)
    mul!(r, A, x, -1, 1)
end

# adds non-zero at i, adds ith column of A to AiQR, and solves resulting ls-problem
function addindex!(x::SparseVector, P::AbstractMatchingPursuit, i::Int)
    addindex!(x, P.AiQR, @view(P.A[:, i]), i)
    return x, P
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
    @. x[ind] = NaN
    for i in ind
        add_column!(P.AiQR, @view(P.A[:,i]))
    end
    ldiv!(x.nzval, P.AiQR, P.b)
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
@inline function factorize!(P::AbstractMatchingPursuit, nzind::Vector{<:Int})
    n = size(P.A, 1)
    k = length(nzind)
    Ai = reshape(@view(P.Ai[1:n*k]), n, k)
    @. Ai = @view(P.A[:, nzind])
    return qr!(Ai) # this still allocates a little memory
end
# ordinary least squares solve
@inline function solve!(P::SP, x::AbstractVector, b::AbstractVector = P.b)
    F = factorize!(P, x)
    ldiv!(x.nzval, F, b)     # optimize all active atoms
end
