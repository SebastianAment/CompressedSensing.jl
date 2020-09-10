# backward algorithms
######################### Backward Regression Algorithm ########################
struct BackwardRegression{T, AT<:AbstractMatrix{T}, B<:AbstractVector{T}, FT} <: AbstractMatchingPursuit{T}
    A::AT
    b::B

    # temporary storage
    r::B # residual
    Ar::B # inner products between measurement matrix and residual
    AiQR::FT # updatable QR factorization of Ai

    # Qa::B # length = k
    # Qb::B
    ε::B # residual norm difference
    lazy::Bool
    rescaling::B # energetic renormalization
end
const BR = BackwardRegression
const BackwardGreedy = BackwardRegression
const BOOMP = BR # a.k.a. Backward Optimized OMP

# lazy = true switches on lazy evaluation of deletion criterion,
# enabled by submodularity of the residual norm
function BR(A::AbstractMatrix, b::AbstractVector; lazy::Bool = true)
    n, m = size(A)
    T = eltype(A)
    r, Ar = zeros(T, n), zeros(T, m)
    # AiQR = UpdatableQR(A)
    AiQR = PUQR(A)
    rescaling = colnorms(A)
    ε = fill(Inf, m)
    return BR(A, b, r, Ar, AiQR, ε, lazy, rescaling)
end

# updates vector of marginal norm increase ε
function update!(P::BR, x::SparseVector)
    # calculate norm of current solution
    P.r .= P.b
    mul!(P.r, P.A, x, -1, 1)
    normr = norm(P.r)

    # reduce all variables to support of x
    A = @view P.A[:, x.nzind]
    ε = @view P.ε[x.nzind]
    n = length(x.nzind)
    y = @view x.nzval[1:n-1] # temporary storage for coefficients of smaller problem

    # we want to check indices with priorly smallest marginal norm increase first
    order = collect(1:n)
    perm = sortperm(ε)
    order = permute!(order, perm)

    minr = Inf # keep track of min marginal norm increase
    for i in order # 1:n unordered
        if ε[i] < minr || isinf(ε[i]) || !P.lazy # flag for lazy evaluation
            a = @view A[:, i]
            Ai = @view A[:, filter(!=(i), 1:length(x.nzind))]
            remove_column!(P.AiQR, i)
            ldiv!(y, P.AiQR, P.b)
            P.r .= P.b
            mul!(P.r, Ai, y, -1, 1)
            add_column!(P.AiQR, a, i)
            ε[i] = norm(P.r) - normr # marginal norm increase
        end
        minr = min(minr, ε[i])
    end
    return ε
end

# to abstract submodular function to use with generic greedy algorithm:
# function f(nzind)
#     A = P.A[:, nzind]
#     norm(P.b) - norm(P.b - A*(A\P.b))
# end

# calculates a solution to a full column rank (not underdetermined)
# linear system Ax = b with the backward greedy algorithm
# TODO: k-sparse stopping criterion
function br(A::AbstractMatrix, b::AbstractVector, δ::Real)
    P = BR(A, b)
    x = P.AiQR \ b
    x = sparse(x)
    n, m = size(A)
    while true
        ε = update!(P, x)
        m, i = findmin(ε) # drop the value which leads to the minimum residual norm
        if m > δ # only delete if we don't cross error threshold
            break
        end
        _dropindex!(x, P.AiQR, i) # i is index into x.nzval, NOT into x
        ldiv!(x.nzval, P.AiQR, b)
    end
    return x
end

# calculates least-squares residual by leaving out the ith column
# y is a temporary vector which stores the coefficients of the lookahead problem
# function br_lookahead!(A, x::AbstractVector, i::Int, y::AbstractVector)
#     a = @view P.A[:, nzi]
#     # a = @view A[:, i]
#     A = @view P.A[:, filter(!=(nzi), x.nzind)]
#     remove_column!(P.AiQR, i)
#     ldiv!(y, P.AiQR, P.b)
#     P.r .= P.b
#     mul!(P.r, A, y, -1, 1)
#     add_column!(P.AiQR, a, i)
# end

################################# LACE #########################################
# Least Absolute Coefficient Elimination
struct LACE{T, AT<:AbstractMatrix{T}, BT<:AbstractVector{T}, FT}
    A::AT # dictionary
    b::BT # target
    r::BT # residual
    AiQR::FT # UpdatableQR
end
function LACE(A::AbstractMatrix, b::AbstractVector)
    r = similar(b)
    n, m = size(A)
    n ≥ m || throw("A needs to be overdetermined but is of size ($n, $m)")
    AiQR = UpdatableQR(A)
    LACE(A, b, r, AiQR)
end

# input x is assumed to be ls-solution with current active set
function update!(L::LACE, x::SparseVector)
    residual!(L, x)
    i = argmin(abs, x.nzval) # choose least absolute coefficient magnitude from current support
    _dropindex!(x, L.AiQR, i) # drops ith atom in active set
    ldiv!(x.nzval, L.AiQR, L.b) # optimize all active atoms
    return x
end

# A is overdetermined linear system, b is target, δ is tolerable residual norm
function lace(A::AbstractMatrix, b::AbstractVector, δ::Real)
    L = LACE(A, b)
    x = spzeros(eltype(A), size(A, 2))
    ldiv!(x.nzval, L.AiQR, L.b)
    while norm(L.r) < δ
        update!(L, x)
    end
    return x
end

################################################################################
# different implementation of backward regression
# deletes most irrelevant atom, if any, as given by energetic subspace norm
# function update!(P::BG, x::SparseVector, δ::Real)
#     nnz(x) > 1 || return false
#     i = deletion_index!(P, x)
#     if P.Ar[i] < δ
#         x[i] = 0
#         dropindex!(P, x, i)
#         ldiv!(x.nzval, P.AiQR, P.b) # optimize all active atoms
#         return true
#     else
#         return false
#     end
# end

# function deletion_index!(P::BR, x::AbstractVector)
#     residual!(P, x)
#     backward_rescaling!(P, x)
#     fudge = 1e-6 # fudge for stability
#     @. P.Ar = abs(P.Ar) / max(P.rescaling, fudge)
#     return argmin(P.Ar)
# end

# calculates energetic norm and inner product with b for all active atoms
# function backward_rescaling!(P::BR, x::SparseVector)
#     P.Ar .= Inf
#     Qa, Qb = zeros(nnz(x)-1), zeros(nnz(x)-1) # pre-allocate this?
#     for (i, nzi) in enumerate(x.nzind) # threads? would need several QRs!
#         remove_column!(P.AiQR, i)
#         a = @view P.A[:, nzi]
#         Q = P.AiQR isa UpdatableQR ? P.AiQR.Q1 : P.AiQR.uqr.Q1
#
#         mul!(Qa, Q', a)
#         mul!(Qb, Q', P.b)
#         P.Ar[nzi] = dot(a, P.b) - dot(Qa, Qb)
#         P.rescaling[nzi] = sum(abs2, a) - sum(abs2, Qa)
#         P.rescaling[nzi] = sqrt(max(P.rescaling[nzi], 0))
#
#         add_column!(P.AiQR, a, i)
#     end
#     return P.rescaling
# end
