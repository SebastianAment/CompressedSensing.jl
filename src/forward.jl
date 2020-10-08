abstract type AbstractStepwiseRegression{T} end
######################### Forward Regression #############################
struct ForwardRegression{T, AT<:AbstractMatrix{T}, B<:AbstractVector{T},
                        V<:AbstractVector{T}, QT<:AbstractMatrix{T}, FT} <: AbstractMatchingPursuit{T}
    A::AT # matrix
    b::B # target
    r::V # residual
    QA::QT # temporary storage
    AiQR::FT # updatable QR factorization of Ai
    δ::V # marginal decrease in objective value
    rescaling::V # OLS rescaling
end
const FR = ForwardRegression
const OrthogonalLeastSquares = ForwardRegression # a.k.a.
const OLS = OrthogonalLeastSquares
const OOMP = OLS # a.k.a. Optimal OMP
const ORMP = OLS # a.k.a order-recursive matching pursuit
const StepwiseRegression = FR

function FR(A::AbstractMatrix, b::AbstractVector)
    n, m = size(A)
    T = eltype(A)
    r, Ar = zeros(T, n), zeros(T, m)
    QA = zeros(T, (n, m))
    # AiQR = UpdatableQR(reshape(A[:, 1], :, 1))
    AiQR = PUQR(reshape(A[:, 1], :, 1))
    remove_column!(AiQR)
    rescaling = colnorms(A)
    δ = fill(-Inf, m)
    FR(A, b, r, QA, AiQR, δ, rescaling)
end

function fr(A::AbstractMatrix, b::AbstractVector;
     max_residual::Real = 0., min_decrease::Real = 0., sparsity::Int = size(A, 2))
     fr(A, b, max_residual, min_decrease, sparsity)
end

# calculates a solution to a potentially underdetermined
# linear system Ax = b with the forward greedy algorithm
# max_ε is the residual norm tolerance
# min_δ is the largest marginal increase in residual norm before the algorithm terminates
# k is the desired sparsity of the solution
# whichever criterion is hit first
function fr(A::AbstractMatrix, b::AbstractVector, max_ε::Real, min_δ::Real,
            k::Int = size(A, 1), x = spzeros(size(A, 2)))
    P = FR(A, b)
    for _ in 1:k
        forward_step!(P, x, max_ε, min_δ)
    end
    return x
end
const ols = fr
const oomp = fr
const ormp = fr
# returns true if foward step was successful, updates x in place
function forward_step!(P::FR, x::SparseVector, max_ε::Real, min_δ::Real)
    nnz(x) < size(P.A, 1) || return false
    residual!(P.r, P.A, x, P.b)
    normr = norm(P.r)
    normr > max_ε || return false
    δ = forward_δ!(P, x)
    max_δ, i = findmax(δ)
    if min_δ < max_δ
        a = @view P.A[:, i]
        addindex!(x, P.AiQR, a, i) # i is index into x.nzval, NOT into x
        ldiv!(x.nzval, P.AiQR, P.b)
        return true
    else
        ldiv!(x.nzval, P.AiQR, P.b)
        return false
    end
end

function forward_δ!(P::FR, x::AbstractVector)
    residual!(P, x)
    mul!(P.δ, P.A', P.r) # δ = Ar = q
    rescaling = ols_rescaling!(P, x)
    @. P.δ = abs(P.δ) / rescaling
    @. P.δ[x.nzind] = 0
    return P.δ
end

function acquisition_index!(P::FR, x::AbstractVector)
    argmax(forward_δ!(P, x))
end

function update!(P::FR, x::AbstractVector)
    nnz(x) < size(P.A, 1) || return false
    i = acquisition_index!(P, x)
    a = @view P.A[:, i]
    addindex!(x, P.AiQR, a, i)
    ldiv!(x.nzval, P.AiQR, P.b) # least-squares solve for active atoms
    return x
end

###############################################################################
# calculates energetic norm of all passive atoms
# √[ |x|^2 - x' * P_I * x] where P_I is the projection onto active column set of A
# Q is assumed to be part of QR-factorization of active set
function ols_rescaling!(P::OLS, x::SparseVector)
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
