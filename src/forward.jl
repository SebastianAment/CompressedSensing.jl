abstract type AbstractStepwiseRegression{T} end
######################### Forward Regression #############################
struct ForwardRegression{T, AT<:AbstractMatOrFac{T}, B<:AbstractVector{T},
                        V<:AbstractVector{T}, QT<:AbstractMatOrFac{T}, FT} <: AbstractMatchingPursuit{T}
    A::AT # matrix
    b::B # target
    r::V # residual
    QA::QT # temporary storage
    AiQR::FT # updatable QR factorization of Ai
    δ²::V # decrease in SQUARED residual norm
    rescaling::V # OLS rescaling
end
const FR = ForwardRegression
const OrthogonalLeastSquares = ForwardRegression # a.k.a.
const OLS = OrthogonalLeastSquares
const OOMP = OLS # a.k.a. Optimal OMP
const ORMP = OLS # a.k.a order-recursive matching pursuit
const StepwiseRegression = FR

FR(A::AbstractMatOrFac, b::AbstractVector, x::SparseVector) = FR(A, b, x.nzind)
function FR(A::AbstractMatOrFac, b::AbstractVector, nzind::AbstractVector{Int} = ones(Int, 0))
    size(A, 1) == length(b) || throw(DimensionMismatch("size(A, 1) = $(size(A, 1)) ≠ $(length(b)) = length(b)"))
    n, m = size(A)
    T = eltype(A)
    r, Ar = zeros(T, n), zeros(T, m)
    QA = zeros(T, (n, m))
    AiQR = UpdatableQR(A[:, nzind])
    rescaling = colnorms(A)
    δ² = fill(-Inf, m)
    FR(A, b, r, QA, AiQR, δ², rescaling)
end

function fr(A::AbstractMatOrFac, b::AbstractVector;
     max_residual::Real = 0., min_decrease::Real = 0., sparsity::Int = size(A, 2))
     fr(A, b, max_residual, min_decrease, sparsity)
end

# calculates a solution to a potentially underdetermined
# linear system Ax = b with the forward greedy algorithm
# max_ε is the residual norm tolerance
# min_δ is the largest marginal increase in residual norm before the algorithm terminates
# k is the desired sparsity of the solution
# whichever criterion is hit first
function fr(A::AbstractMatOrFac, b::AbstractVector, max_ε::Real, min_δ::Real,
            k::Int = size(A, 1), x = spzeros(size(A, 2)))
    P = FR(A, b)
    for i in 1:k
        forward_step!(P, x, max_ε, min_δ) || break
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
    δ² = forward_δ!(P, x)
    max_δ², i = findmax(δ²)
    if min_δ^2 < max_δ²
        a = @view P.A[:, i]
        addindex!(x, P.AiQR, a, i) # i is index into x.nzval, NOT into x
        ldiv!!(x.nzval, P.AiQR, P.b, P.r)
        return true
    else
        ldiv!!(x.nzval, P.AiQR, P.b, P.r)
        return false
    end
end

function forward_δ!(P::FR, x::AbstractVector)
    residual!(P, x)
    mul!(P.δ², P.A', P.r) # δ = Ar = q
    rescaling = ols_rescaling!(P, x)
    @. P.δ² = P.δ²^2 / rescaling
    @. P.δ²[x.nzind] = 0
    return P.δ²
end

function acquisition_index!(P::FR, x::AbstractVector)
    argmax(forward_δ!(P, x))
end

function update!(P::FR, x::AbstractVector)
    nnz(x) < size(P.A, 1) || return false
    i = acquisition_index!(P, x)
    a = @view P.A[:, i]
    addindex!(x, P.AiQR, a, i)
    ldiv!!(x.nzval, P.AiQR, P.b, P.r) # least-squares solve for active atoms, reuses P.r for intermediate result
    return x
end

###############################################################################
# calculates squared energetic norm of all passive atoms
# |x|^2 - x' * P_I * x where P_I is the projection onto active column set of A
# Q is assumed to be part of QR-factorization of active set
function ols_rescaling!(P::OLS, x::SparseVector)
    Q = P.AiQR.Q
    # @. P.QA = P.A
    # lmul!(Q, P.QA)
    mul!(P.QA, Q', P.A)
    sum!(abs2, P.rescaling', P.A) # unnecessary if P.A is normalized
    QA = @view P.QA[1:nnz(x), :]
    k, m = size(QA)
    for j in 1:m
        @simd for i in 1:k
            @inbounds P.rescaling[j] -= QA[i, j]^2
        end
    end
    return P.rescaling
end
