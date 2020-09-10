######################### Forward Regression #############################
struct ForwardRegression{T, AT<:AbstractMatrix{T}, B<:AbstractVector{T},
                                    FT} <: AbstractMatchingPursuit{T}
    A::AT
    b::B

    # temporary storage
    r::B # residual
    qa::B # temporary
    AiQR::FT # updatable QR factorization of Ai

    ε::B # marginal decrease in objective value
    lazy::Bool
end
const FR = ForwardRegression
const OrthogonalLeastSquares = ForwardRegression # a.k.a.
const OLS = OrthogonalLeastSquares
const OOMP = OLS # a.k.a. Optimal OMP
const ORMP = OLS # a.k.a order-recursive matching pursuit

function FR(A::AbstractMatrix, b::AbstractVector; lazy::Bool = true)
    n, m = size(A)
    T = eltype(A)
    r, Ar = zeros(T, n), zeros(T, m)
    qa = zeros(T, n)
    # AiQR = UpdatableQR(reshape(A[:, 1], :, 1))
    AiQR = PUQR(reshape(A[:, 1], :, 1))
    remove_column!(AiQR)
    rescaling = colnorms(A)
    ε = fill(Inf, m)
    FR(A, b, r, qa, AiQR, ε, lazy)
end
# approximately solves Ax = b with error tolerance δ it at most k steps
function fr(A::AbstractMatrix, b::AbstractVector, k::Int = size(A, 1))
    fr(A, b, size(A, 1)*eps(eltype(b)), k)
end

function fr(A::AbstractMatrix, b::AbstractVector, δ::Real, k::Int = size(A, 1),
                                                        x = spzeros(size(A, 2)))
    δ ≥ 0 || throw("δ = $δ has to be non-negative")
    P = FR(A, b)
    while true
        ε = update!(P, x)
        i = argmin(ε) # marginal decrease in objective
        #add non-zero index to active set
        x[i] = NaN
        # efficiently update qr factorization using Givens rotations
        qr_i = findfirst(==(i), x.nzind) # index in qr where new atom should be added
        a = @view P.A[:, i]
        add_column!(P.AiQR, a, qr_i)
        # least-squares solve for active atoms
        ldiv!(x.nzval, P.AiQR, P.b)

        (norm(residual!(P, x)) ≥ δ && !(nnz(x) ≥ k)) || break
    end
    return x
end

const ols = fr
const oomp = fr
const ormp = fr

function update!(P::FR, x::AbstractVector)
    n, m = size(P.A)

    # calculate current residual, necessary to evaluate marginal change
    residual!(P, x)

    # due to submodularity, only necessary to compute fully in first iteration
    # and only if P.ε has not been evaluated
    if (nnz(x) == 0 && all(==(Inf), P.ε)) # || !lazy # need to compute ols_rescaling!
        mul!(P.ε, P.A', P.r)
        for i in 1:m
            P.ε[i] = -abs(P.ε[i]) / norm(@view P.A[:, i]) # if we assume A is normalized, don't need this
        end
        @. P.ε[x.nzind] = 0
        return P.ε
    end

    # # TODO: we want to check indices with priorly smallest marginal norm increase first
    # order = collect(1:n)
    # perm = sortperm(ε)
    # order = permute!(order, perm)

    passive = ones(Bool, m) # pre-allocate?
    passive[x.nzind] .= false

    ε = P.ε
    minr = Inf # tracking best marginal change in residual norm
    for i = 1:m # thread-safe?
        passive[i] || continue # if ith atom is active, skip evaluation of ε
        a = @view P.A[:, i]
        if ε[i] < minr || isinf(ε[i]) || !P.lazy # flag for lazy evaluation
            ε[i] = -abs(dot(a, P.r)) / ols_rescaling(P, a)
        end
        minr = min(minr, ε[i])
    end
    @. ε[x.nzind] = 0
    return ε #@view ε[passive] # cannot add already active atoms
end

# calculates energetic norm of all passive atoms
# √[ |x|^2 - x' * P_I * x] where P_I is the projection onto active column set of A
# Q is assumed to be part of QR-factorization of active set
function ols_rescaling(P::OLS, a::AbstractVector)
    nnz = size(P.AiQR, 2) # number of current nonzeros
    scaling = sum(abs2, a)
    if nnz == 0
        return scaling
    else
        Q = P.AiQR isa UpdatableQR ? P.AiQR.Q1 : P.AiQR.uqr.Q1
        qa = @view P.qa[1:nnz]
        mul!(qa, Q', a)
        scaling -= sum(abs2, qa)
        return sqrt(max(scaling, 0))
    end
end

################################################################################
# OLD FR implemention, not taking advantage of submodularity
# function update!(P::FR, x::AbstractVector)
#     nnz(x) < size(P.A, 1) || return false
#     i = acquisition_index!(P, x)
#     # add non-zero index to active set
#     x[i] = NaN
#     # efficiently update qr factorization using Givens rotations
#     qr_i = findfirst(==(i), x.nzind) # index in qr where new atom should be added
#     a = @view P.A[:, i]
#     add_column!(P.AiQR, a, qr_i)
#     # least-squares solve for active atoms
#     ldiv!(x.nzval, P.AiQR, P.b)
#     return true
# end
#
# acquisition index
# function acquisition_index!(P::FR, x::AbstractVector)
#     residual!(P, x)
#     mul!(P.Ar, P.A', P.r) # Ar = q
#     ols_rescaling!(P, x)
#     stability_fudge = 1e-6 # fudge for stability
#     @. P.Ar = abs(P.Ar) / max(P.rescaling, stability_fudge)
#     @. P.Ar[x.nzind] = 0
#     return argmax(P.Ar)
# end
#
# # calculates energetic norm of all passive atoms
# # √[ |x|^2 - x' * P_I * x] where P_I is the projection onto active column set of A
# # Q is assumed to be part of QR-factorization of active set
# function ols_rescaling!(P::OLS, x::SparseVector)
#     Q = P.AiQR isa UpdatableQR ? P.AiQR.Q1 : P.AiQR.uqr.Q1
#     QA = @view P.QA[1:nnz(x), :]
#     mul!(QA, Q', P.A)
#     sum!(abs2, P.rescaling', P.A) # unnecessary if P.A is normalized
#     k, m = size(QA)
#     for j in 1:m
#         @simd for i in 1:k
#             @inbounds P.rescaling[j] -= QA[i,j]^2
#         end
#     end
#     @. P.rescaling = sqrt(max(P.rescaling, 0))
# end
