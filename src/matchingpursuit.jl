abstract type AbstractMatchingPursuit{T} <: Update{T} end

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
    P! = MP(A, b, k)
    for i in 1:k
        P!(x)
    end
    x
end

###################### Orthogonal Matching Pursuit #############################
# could extend: preconditioning, non-negativity constraint
struct OrthogonalMatchingPursuit{T, AT<:AbstractMatrix{T}, B<:AbstractVector{T},
                                            FT} <: AbstractMatchingPursuit{T}
    A::AT
    b::B
    k::Int # maximum number of non-zeros

    # temporary storage
    r::B # residual
    Ar::B # inner products between measurement matrix and residual
    AiQR::FT # updatable QR factorization
end
const OMP = OrthogonalMatchingPursuit
function OMP(A::AbstractMatrix, b::AbstractVector, k::Integer = size(A, 1))
    n, m = size(A)
    T = eltype(A)
    r, Ar = zeros(T, n), zeros(T, m)
    # AiQR = UpdatableQR(reshape(A[:, 1], :, 1))
    AiQR = PUQR(reshape(A[:, 1], :, 1))
    remove_column!(AiQR) # start with empty factorization
    OMP(A, b, k, r, Ar, AiQR)
end

function update!(P::OMP, x::AbstractVector = spzeros(size(P.A, 2)))
    residual!(P, x)
    i = argmaxinner!(P)
    ∉(i, x.nzind) || return x
    # add non-zero index to active set
    x[i] = NaN
    # update qr factorization using Givens rotations
    qr_i = findfirst(==(i), x.nzind) # index in qr where new atom should be added
    a = @view P.A[:, i]
    add_column!(P.AiQR, a, qr_i)
    # least-squares solve for active atoms
    ldiv!(x.nzval, P.AiQR, P.b)
    return x
end

# approximately solves Ax = b with error tolerance δ it at most k steps
function omp(A::AbstractMatrix, b::AbstractVector, δ::Real, k::Int = size(A, 1))
    δ ≥ 0 || throw("δ = $δ has to be non-negative")
    P = OMP(A, b)
    x = spzeros(size(A, 2))
    for i in 1:k
        P(x)
        norm(residual!(P, x)) ≥ δ || break
    end
    return x
end
# calculates k-sparse approximation to Ax = b via orthogonal matching pursuit
omp(A::AbstractMatrix, b::AbstractVector, k::Int) = omp(A, b, eps(eltype(A)), k)

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
const SSP = SP

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
ssp_index!(P::SP, k::Int = P.k) = ompr_index!(P, k)
ssp_acquisition!(P::SP, x, k::Int = P.k) = ompr_acquisition!(P, x, k)

# TODO: could pre-allocate nz arrays to be of length 2K
# TODO: could add efficient qr updating
function update!(P::SP, x::AbstractVector = spzeros(size(P.A, 2)), ε::Real = 0.)
    if nnz(x) < P.k
        ssp_acquisition!(P, x, P.k-nnz(x))
    end
    nnz(x) == P.k || throw("nnz(x) = $(nnz(x)) ≠ $(P.k) = k")
    ssp_acquisition!(P, x)
    i = partialsortperm(abs.(x.nzval), 1:nnz(x)-P.k) # find the smallest atoms
    @. x.nzval[i] = 0

    droptol!(x, ε)
    solve!(P, x) # optimize all active atoms
end

# calculates k-sparse approximation to Ax = b via subspace pursuit
# could also stop if indices are same between iterations
function sp(A::AbstractMatrix, b::AbstractVector, k::Int, δ::Real = 1e-12; maxiter = 16)
    P! = SP(A, b, k)
    x = spzeros(size(A, 2))
    for i in 1:maxiter
        P!(x)
        if norm(residual!(P!, x)) < δ # if we found a solution with the required sparsity we're done
            break
        end
    end
    return x
end
const ssp = sp
####################### Noiseless Relevance Pursuit ############################
# Thought:
# As long as we are not deleting, excluded irrelevant atoms cannot become relevant
# noiseless limit of greedy sbl algorithm
struct RelevanceMatchingPursuit{T, AT<:AbstractMatrix{T}, B<:AbstractVector{T},
                                    FT} <: AbstractMatchingPursuit{T}
    A::AT
    b::B

    # temporary storage
    r::B # residual
    Ar::B # inner products between measurement matrix and residual
    QA::AT # (k, m)
    AiQR::FT # updatable QR factorization of Ai

    rescaling::B # energetic renormalization
    rescale::Bool # toggles rescaling by energetic norm
end
const RMP = RelevanceMatchingPursuit

function RMP(A::AbstractMatrix, b::AbstractVector; rescale::Bool = true)
    n, m = size(A)
    T = eltype(A)
    r, Ar = zeros(T, n), zeros(T, m)
    QA = zeros(T, (n, m))
    AiQR = UpdatableQR(reshape(A[:, 1], :, 1))
    # AiQR = PUQR(reshape(A[:, 1], :, 1))
    remove_column!(AiQR)
    rescaling = sqrt.(sum(abs2, A, dims = 1))
    rescaling = reshape(rescaling, :)
    RMP(A, b, r, Ar, QA, AiQR, rescaling, rescale)
end

# calculates energetic norm of all passive atoms
function rmp_passive_rescaling!(P::RMP, x::SparseVector)
    Q = P.AiQR isa UpdatableQR ? P.AiQR.Q1 : P.AiQR.uqr.Q1
    QA = @view P.QA[1:nnz(x), :]
    mul!(QA, Q', P.A)
    sum!(abs2, P.rescaling', P.A) # unnecessary if P.A is normalized
    k, m = size(QA)
    for j in 1:m
        @simd for i in 1:k
            @inbounds P.rescaling[j] -= QA[i,j]^2
        end
    end
    @. P.rescaling = sqrt(max(P.rescaling, 0))
end

# acquisition index
function rmp_acquisition_index!(P::RMP, x::AbstractVector)
    residual!(P, x)
    mul!(P.Ar, P.A', P.r) # Ar = q
    if P.rescale
        rmp_passive_rescaling!(P, x)
        fudge = 1e-6 # fudge for stability
        @. P.Ar = abs(P.Ar) / max(P.rescaling, fudge)
    else
        @. P.Ar = abs(P.Ar)
    end
    @. P.Ar[x.nzind] = 0
    return argmax(P.Ar)
end

function add_relevant!(P::RMP, x::AbstractVector, δ::Real)
    nnz(x) < size(P.A, 1) || return false
    i = rmp_acquisition_index!(P, x)
    residual!(P, x)
    # i = argmaxinner!(P)
    if P.Ar[i] > δ
        # add non-zero index to active set
        x[i] = NaN
        # efficiently update qr factorization using Givens rotations
        qr_i = findfirst(==(i), x.nzind) # index in qr where new atom should be added
        a = @view P.A[:, i]
        add_column!(P.AiQR, a, qr_i)
        # least-squares solve for active atoms
        ldiv!(x.nzval, P.AiQR, P.b)
        return true
    else
        return false
    end
end

# calculates energetic norm and inner product with b for all active atoms
function rmp_check_relevance!(P::RMP, x::SparseVector)
    P.Ar .= Inf
    Qa = zeros(nnz(x)-1)
    Qb = zeros(nnz(x)-1)
    for (i, nzi) in enumerate(x.nzind) # threads?
        remove_column!(P.AiQR, i)
        a = @view P.A[:, nzi]
        # Q = P.AiQR.uqr.Q1
        Q = P.AiQR isa UpdatableQR ? P.AiQR.Q1 : P.AiQR.uqr.Q1
        mul!(Qa, Q', a)
        if P.rescale
            P.rescaling[nzi] = sum(abs2, a) - sum(abs2, Qa)
            P.rescaling[nzi] = sqrt(max(P.rescaling[nzi], 0))
        end
        mul!(Qb, Q', P.b)
        P.Ar[nzi] = dot(a, P.b) - dot(Qa, Qb)
        add_column!(P.AiQR, a, i)
    end
    P.rescaling
end

function rmp_deletion_index!(P::RMP, x::AbstractVector)
    residual!(P, x)
    rmp_check_relevance!(P, x)
    if P.rescale
        fudge = 1e-6 # fudge for stability
        @. P.Ar = abs(P.Ar) / max(P.rescaling, fudge)
    else
        @. P.Ar = abs(P.Ar)
    end
    return argmin(P.Ar)
end

# deletes most irrelevant atom, if any, as given by energetic subspace norm
function delete_irrelevant!(P::RMP, x::SparseVector, δ::Real)
    nnz(x) > 1 || return false
    i = rmp_deletion_index!(P, x)
    if P.Ar[i] < δ
        x[i] = 0
        # dropzeros!(P, x) # this updates the index set and factorization
        dropindex!(P, x, i)
        ldiv!(x.nzval, P.AiQR, P.b) # optimize all active atoms
        return true
    else
        return false
    end
end

# TODO: benchmark! and test against other algs
# provide option which gives k-sparse result
function rmp(A::AbstractMatrix, b::AbstractVector, δ::Real = 1e-12,
            x = spzeros(size(A, 2)); rescale::Bool = false)
    P = RMP(A, b, rescale = rescale)
    nzind = copy(x.nzind)
    n = size(A, 1)
    # for _ in 1:maxiter
    for i in 1:n # acquisition stage (equivalent to OLS)
        add_relevant!(P, x, δ) || break
    end
    for i in 1:n # deletion stage (equivalent to backward greedy)
        delete_irrelevant!(P, x, δ) || break
    end
    # nzind != x.nzind || break # if the index set hasn't changed
    # copy!(nzind, x.nzind)
    # end
    return x
end

# function rmp(A::AbstractMatrix, b::AbstractVector, δ::Real = 1e-12,
#             x = spzeros(size(A, 2)); maxiter = size(A, 1), rescale::Bool = false)
#     P = RMP(A, b, rescale = rescale)
#     for i in 1:maxiter
#         add_relevant!(P, x, δ) || break
#         for j in 1:maxiter # deletion stage
#             delete_irrelevant!(P, x, δ) || break
#         end
#     end
#     return x
# end

function dropindex!(P::RMP, x::SparseVector, i::Int)
    j = findfirst(==(i), x.nzind)
    if !isnothing(j)
		deleteat!(x.nzind, j)
		deleteat!(x.nzval, j)
        remove_column!(P.AiQR, j)
    end
    return P
end
function SparseArrays.droptol!(P::RMP, x::SparseVector, tol::Real)
    for i in reverse(eachindex(x.nzind)) # reverse is necessary to not mess up indexing into QR factorization
        if abs(x.nzval[i]) ≤ tol
            remove_column!(P.AiQR, i)
        end
    end
    droptol!(x, tol)
end

function SparseArrays.dropzeros!(P::RMP, x::SparseVector)
    droptol!(P, x, 0)
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
    if nnz(x) < P.k # make sure support set is of size k
        @. x = 0
        dropzeros!(x)
        ind = sample(1:length(x), P.k, replace = false) # or could initialize with max inner products
        @. x[ind] = NaN
        for i in ind
            add_column!(P.AiQR, @view(P.A[:,i]))
        end
        ldiv!(x.nzval, P.AiQR, P.b)
    end
    nnz(x) == P.k || throw("nnz(x) = $(nnz(x)) ≠ $(P.k) = k")

    residual!(P, x)
    copy!(P.Ar, x)
    mul!(P.Ar, P.A', P.r, η, 1)
    begin
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
# l is cardinality of maximum replacement per iteration, l = k corresponds to ssp
function ompr(A::AbstractMatrix, b::AbstractVector, k::Int, δ::Real = 1e-12,
                                x = spzeros(size(A, 2)); maxiter = size(A, 1))
    P = OMPR(A, b, k)
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

############################# Optimized OMP ####################################
# approximately solves Ax = b with error tolerance δ it at most k steps
function oomp(A::AbstractMatrix, b::AbstractVector, k::Int = size(A, 1))
    oomp(A, b, size(A, 1)*eps(eltype(b)), k)
end
function oomp(A::AbstractMatrix, b::AbstractVector, δ::Real, k::Int = size(A, 1),
                                                        x = spzeros(size(A, 2)))
    δ ≥ 0 || throw("δ = $δ has to be non-negative")
    P = RMP(A, b, rescale = true)
    for i in 1:k # acquisition stage of RMP (equivalent to OLS, OOMP, ORMP)
        add_relevant!(P, x, zero(eltype(b)))
        norm(residual!(P, x)) ≥ δ || break
    end
    return x
end
# const OLS = OOMP # a.k.a. orthogonal least-squares
# const ORMP = OOMP # a.k.a order-recursive matching pursuit
const ols = oomp
const ormp = oomp

############################## Backward Greedy #################################
# TODO: calculates a solution to a full column rank (and not underdetermined)
# linear system Ax = b with the backward greedy algorithm
# function backward_greedy(A, b)
#     AF = UpdatableQR(A)
#     x = AF \ b
#
#     for i in 1:size(A, 2)
#         remove_column!(AF, i)
#         add_column(AF, )
#     end
#
# end

####################### Matching Pursuit Helpers ###############################
# calculates residual of
@inline function residual!(P::AbstractMatchingPursuit, x::AbstractVector)
    residual!(P.r, P.A, x, P.b)
end
@inline function residual!(r::AbstractVector, A::AbstractMatrix, x::AbstractVector, b::AbstractVector)
    copyto!(r, b)
    mul!(r, A, x, -1, 1)
end
########################### acquisition criteria ###############################
# WARNING: need to compute residual first
# returns index of atom with largest dot product with residual
# a.k.a omp_index
@inline function argmaxinner!(P::AbstractMatchingPursuit) # 0 alloc
    mul!(P.Ar, P.A', P.r)
    @. P.Ar = abs(P.Ar)
    argmax(P.Ar)
end

# returns indices of k atoms with largest inner products with residual
@inline function ompr_index!(P::AbstractMatchingPursuit, k::Int = P.k)
    mul!(P.Ar, P.A', P.r)
    @. P.Ar = abs(P.Ar)
    partialsortperm(P.Ar, 1:k, rev = true)
end

function ompr_acquisition!(P::AbstractMatchingPursuit, x, k::Int = P.k)
    residual!(P, x)
    i = ompr_index!(P, k)
    @. x[i] = NaN
    solve!(P, x)
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
@inline function solve!(P::AbstractMatchingPursuit, x::AbstractVector, b::AbstractVector = P.b)
    F = factorize!(P, x)
    ldiv!(x.nzval, F, b)     # optimize all active atoms
end
