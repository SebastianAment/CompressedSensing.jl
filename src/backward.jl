# backward algorithms
######################### Backward Regression Algorithm ########################
struct BackwardRegression{T, AT<:AbstractMatrix{T}, B<:AbstractVector{T}, FT} <: AbstractMatchingPursuit{T}
    A::AT # matrix
    b::B # target
    r::B # residual
    AiQR::FT # updatable QR factorization of Ai
    δ::B # residual norm difference
end
const BR = BackwardRegression
const BackwardGreedy = BackwardRegression
const BOOMP = BR # a.k.a. Backward Optimized OMP

function BR(A::AbstractMatrix, b::AbstractVector)
    n, m = size(A)
    r = zeros(eltype(A), m)
    # AiQR = UpdatableQR(A)
    AiQR = PUQR(A)
    rescaling = colnorms(A)
    δ = fill(-Inf, m)
    return BR(A, b, r, AiQR, δ)
end

# calculates a solution to a full column rank (not underdetermined)
# linear system Ax = b with the backward greedy algorithm
# max_ε is the residual norm tolerance
# max_δ is the largest marginal increase in residual norm before the algorithm terminates
# k is the desired sparsity of the solution
# whichever criterion is hit first
function br(A::AbstractMatrix, b::AbstractVector, max_ε::Real, max_δ::Real, k::Int)
    P = BR(A, b)
    x = P.AiQR \ b
    x = sparse(x)
    n, m = size(A)
    for _ in m:-1:k+1
        backward_step!(P, x, max_ε, max_δ) || break
    end
    return x
end

# keyword version
function br(A::AbstractMatrix, b::AbstractVector;
     max_residual::Real = Inf, max_increase::Real = Inf, sparsity::Int = 0)
     br(A, b, max_residual, max_increase, sparsity)
end

function backward_step!(P::Union{FR, BR}, x::SparseVector, max_ε::Real, max_δ::Real)
    nnz(x) > 0 || return false
    residual!(P.r, P.A, x, P.b)
    normr = norm(P.r)
    δ = backward_δ!(P, x, normr)
    min_δ, i = findmin(δ) # drop the atom that leads to the minimum increase of the residual norm
    if min_δ + normr < max_ε && min_δ < max_δ
        _dropindex!(x, P.AiQR, i) # i is index into x.nzval, NOT into x
        ldiv!(x.nzval, P.AiQR, P.b)
        return true
    else
        ldiv!(x.nzval, P.AiQR, P.b)
        return false
    end
end

# updates vector of marginal norm increase δ
# WARNING: assumes P.r == P.b - P.A*x
# or normr = norm(P.b - P.A*x)
function backward_δ!(P::Union{FR, BR}, x::SparseVector, normr = norm(P.r))
    # reduce all variables to support of x
    A = @view P.A[:, x.nzind]
    δ = @view P.δ[x.nzind]
    n = length(x.nzind)
    # y = @view x.nzval[1:n-1] # temporary storage for coefficients of smaller problem
    y = similar(x.nzval, n-1)
    for i in 1:n # could be parallelized if y and P.r are separate for each thread
        a = @view A[:, i]
        Ai = @view A[:, filter(!=(i), 1:length(x.nzind))]
        remove_column!(P.AiQR, i)
        ldiv!(y, P.AiQR, P.b)
        P.r .= P.b
        mul!(P.r, Ai, y, -1, 1)
        δ[i] = norm(P.r) - normr # marginal norm increase
        add_column!(P.AiQR, a, i)
    end
    return δ
end

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
    L.r .= L.b
    mul!(L.r, L.A, x, -1, 1)
    # normr = norm(L.r)
    i = argmin(abs, x.nzval) # choose least absolute coefficient magnitude from current support
    _dropindex!(x, L.AiQR, i) # drops ith atom in active set
    ldiv!(x.nzval, L.AiQR, L.b) # optimize all active atoms
    return x
end

# A is overdetermined linear system, b is target, ε is tolerable residual norm
function lace(A::AbstractMatrix, b::AbstractVector, ε::Real, k::Int)
    n, m = size(A)
    L = LACE(A, b)
    x = sparse(ones(eltype(A), size(A, 2)))
    ldiv!(x.nzval, L.AiQR, L.b)
    for _ in m:-1:k+1
        update!(L, x)
        if norm(L.r) > ε
            break
        end
    end
    return x
end

########################### probabilistic bounds ###############################
# calculates bound on maximum inner product of
# standard normal random vector with n l2-normalized vectors
# function normal_infbound(p, n)
#     η = 2*(1-p)*sqrt(π/log(n))
#     return sqrt(2*(1+η)*log(n))
# end
#
# # calculates a bound on the maximum absolute value
# # of n standard normal random variables
# # which is satisfied with probability p
# function br_maxbound(p, n)
#    d = (1+p^(1/n)) / 2
#    return erfinv(d)
# end
# # calculates a bound on the minimum absolute value
# # of n standard normal random variables
# # which is satisfied with probability p
# function br_minbound(p, n)
#    d = 1 - (1-p)^(1/n) / 2
#    return erfinv(d)
# end
#
# # computes both existing infbound, and the new min-max-bound
# # p is probability with which the bound should hold
# # n is size of linear system
# # k is sparsity level of true solution
# function br_compare_gaussian_bounds(p, n, k)
#     dnew = br_maxbound(√p, k) + br_minbound(√p, n-k)
#     dold = √2 * inner_infbound(p, (n-k)*k)
#     return dnew, dold
# end
