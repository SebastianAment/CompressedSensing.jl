# backward algorithms
######################### Backward Regression Algorithm ########################
struct BackwardRegression{T, AT<:AbstractMatOrFac{T}, B<:AbstractVector{T},
                        V<:AbstractVector{T}, FT, N} <: AbstractMatchingPursuit{T}
    A::AT # matrix
    b::B # target
    r::V # residual
    AiQR::FT # updatable QR factorization of Ai
    δ²::V # increase in SQUARED residual norm
    isfast::N # toggles fast implementation
end
const BR = BackwardRegression
const BackwardGreedy = BackwardRegression
const BOOMP = BR # a.k.a. Backward Optimized OMP

function BR(A::AbstractMatOrFac, b::AbstractVector; isfast::Bool = true)
    n, m = size(A)
    r = zeros(eltype(A), n)
    AiQR = UpdatableQR(A)
    rescaling = colnorms(A)
    δ² = fill(-Inf, m)
    return BR(A, b, r, AiQR, δ², Val(isfast))
end

# calculates a solution to a full column rank (not underdetermined)
# linear system Ax = b with the backward greedy algorithm
# max_ε is the residual norm tolerance
# max_δ is the largest marginal increase in residual norm before the algorithm terminates
# k is the desired sparsity of the solution
# whichever criterion is hit first
function br(A::AbstractMatOrFac, b::AbstractVector, max_ε::Real, max_δ::Real, k::Int; isfast::Bool = true)
    m = size(A, 2)
    P = BR(A, b, isfast = isfast)
    x = sparsevec(1:m, P.AiQR \ b)
    for _ in m:-1:k+1
        backward_step!(P, x, max_ε, max_δ, P.isfast) || break
    end
    return x
end

# keyword version
function br(A::AbstractMatOrFac, b::AbstractVector; max_residual::Real = Inf,
            max_increase::Real = Inf, sparsity::Int = 0, isfast::Bool = true)
     br(A, b, max_residual, max_increase, sparsity, isfast = isfast)
end

function update!(P::BR, x::SparseVector)
    backward_step!(P, x, Inf, Inf)
end

function backward_step!(P::Union{FR, BR}, x::SparseVector, max_ε::Real, max_δ::Real,
                                                        isfast::Val = Val(true))
    nnz(x) > 0 || return false
    residual!(P.r, P.A, x, P.b)
    normr = norm(P.r)
    δ² = isfast isa Val{true} ? backward_δ!(P, x) : naive_backward_δ!(P, x)
    min_δ², i = findmin(δ²) # drop the atom that leads to the minimum increase of the residual norm
    new_norm = sqrt(min_δ² + normr^2) # since min_δ is the squared marginal increase in norm
    if new_norm < max_ε && min_δ² < max_δ^2
        _dropindex!(x, P.AiQR, i) # i is index into x.nzval, NOT into x
        ldiv!!(x.nzval, P.AiQR, P.b, P.r)
        return true
    else
        ldiv!!(x.nzval, P.AiQR, P.b, P.r)
        return false
    end
end

function get_gamma(P::Union{FR, BR})
    F = P.AiQR
    AA = F.R \ Matrix(F.R' \ I)
    ip = invperm(F.perm)
    return γ = diag(AA)[ip]
end

# updates vector of square root of increase in SQUARED norm δ²
# this is done for compatability with RelevanceMatchingPursuit,
# since |r_{-i}|^2 - |r|^2 = |<φ, r>|^2 / |ψ|^2
function backward_δ!(P::Union{FR, BR}, x::SparseVector)
    γ = get_gamma(P)
    @. P.δ²[x.nzind] = x.nzval^2  / γ
    # return P.δ²[x.nzind]
end

# WARNING: assumes P.r == P.b - P.A*x when function is called
# or normr = norm(P.b - P.A*x)
function naive_backward_δ!(P::Union{FR, BR}, x::SparseVector, normr = norm(P.r))
    # reduce all variables to support of x
    A = @view P.A[:, x.nzind]
    δ² = @view P.δ²[x.nzind]
    n = length(x.nzind)
    # y = @view x.nzval[1:n-1] # temporary storage for coefficients of smaller problem
    y = similar(x.nzval, n-1)
    for i in 1:n # could be parallelized if y and P.r are separate for each thread
        a = @view A[:, i]
        Ai = @view A[:, filter(!=(i), 1:length(x.nzind))]
        remove_column!(P.AiQR, i)
        ldiv!!(y, P.AiQR, P.b, P.r)
        P.r .= P.b
        mul!(P.r, Ai, y, -1, 1)
        δ²[i] = norm(P.r)^2 - normr^2 # squared residual norm increase
        add_column!(P.AiQR, a, i)
    end
    return δ²
end

################################################################################
# see "An Efficient Implementation of the Backward Greedy Algorithm for Sparse Signal Reconstruction"
# WARNING: potentially more susceptible to ill-conditioned systems
# except for research purposes, the isfast = true option of BackwardRegression should be used
# IDEA: could implement corresponding forward regression which keeps track of AA⁻¹ instead of QR
struct FastBackwardRegression{T, AT<:AbstractMatOrFac{T}, B<:AbstractVector{T},
            V<:AbstractVector{T}, AAT<:AbstractMatOrFac{T}} <: AbstractMatchingPursuit{T}
    A::AT # matrix
    b::B # target
    r::V # residual
    AA⁻¹::AAT # stores AA⁻¹ corresonding to active set in upper left block
    Ab::V # stores A' * b
    δ²::V # increase in SQUARED residual norm
end
const FBR = FastBackwardRegression
function FBR(A::AbstractMatOrFac, b::AbstractVector)
    r = similar(b)
    return FBR(A, b, r, qr(A))
end
# utilizing the already existing QR-factorization
function FBR(A::AbstractMatOrFac, b::AbstractVector, r::AbstractVector, F::Factorization)
    n, m = size(A)
    AA = F.R \ Matrix(F.R' \ I(m))
    Ab = A'b
    δ² = zeros(m)
    return FBR(A, b, r, AA, Ab, δ²)
end
function FBR(A::AbstractMatOrFac, b::AbstractVector, r::AbstractVector, F::UpdatableQR)
    n, m = size(F)
    E = F.uqr
    AA = E.R1 \ Matrix(E.R1' \ I(m))
    ip = invperm(F.perm)
    AA = AA[ip, ip]
    Ab = A'b
    δ² = zeros(m)
    return FBR(A, b, r, AA, Ab, δ²)
end
# build FBR object from existing ForwardRegression object
function FBR(P::FR)
    FBR(P.A, P.b, P.r, P.AiQR)
end
# TODO: has some overlap with BackwardRegression, which could be consolidated
function fbr(A::AbstractMatOrFac, b::AbstractVector;
     max_residual::Real = Inf, max_increase::Real = Inf, sparsity::Int = 0)
     fbr(A, b, max_residual, max_increase, sparsity)
end

function fbr(A::AbstractMatOrFac, b::AbstractVector, max_ε::Real, max_δ::Real, k::Int)
    m = size(A, 2)
    P = FBR(A, b)
    x = sparsevec(1:m, P.AA⁻¹ * P.Ab)
    for i in m:-1:k+1
        backward_step!(P, x, max_ε, max_δ) || break
    end
    return x
end

function backward_step!(P::FBR, x::SparseVector, max_ε::Real, max_δ::Real)
    nnz(x) > 0 || return false
    normr = norm(residual!(P.r, P.A, x, P.b))
    δ² = backward_δ!(P, x)
    min_δ², i = findmin(δ²) # drop the atom that leads to the minimum increase of the residual norm
    if min_δ² + normr^2 < 0
        println(min_δ² + normr^2)
        println([min_δ², normr^2])
        throw("numerical instability encountered in backward step")
    end
    new_norm = sqrt(min_δ² + normr^2) # since δ is the marginal increase of squared norm
    if new_norm < max_ε && min_δ² < max_δ^2
        _dropindex!(x, P, i) # i is index into x.nzval, NOT into x
        _solve!(P, x)
        return true
    else
        _solve!(P, x)
        return false
    end
end

# solves the least squares problem constrained to non-zero elements of x
# WARNING: assumes P.AA⁻¹ is updated correctly
function _solve!(P::FBR, x::SparseVector)
    AA⁻¹, Ab = @views P.AA⁻¹[1:nnz(x), 1:nnz(x)], P.Ab[x.nzind]
    mul!(x.nzval, AA⁻¹, Ab)
    return x
end

function backward_δ!(P::FBR, x::SparseVector)
    m = nnz(x)
    AA⁻¹ = @views P.AA⁻¹[1:m, 1:m]
    δ² = @view P.δ²[1:m]
    γ = @view AA⁻¹[diagind(AA⁻¹)]
    @. δ² = x.nzval^2 / γ # since x = AA⁻¹ * A' * P.b
end

# drops the ith index of x and updates the matrices AA⁻¹, AA⁻¹A accordingly
function _dropindex!(x::SparseVector, P::FBR, i::Int)
    j = x.nzind[i]
    m = nnz(x)
    deleteat!(x.nzind, i)
    deleteat!(x.nzval, i)
    A, AA⁻¹ = @views P.A[:, x.nzind], P.AA⁻¹[1:m, 1:m]
    active = 1:m .!= i
    G, g, γ = AA⁻¹[active, active], AA⁻¹[i, active], AA⁻¹[i, i]
    @. P.AA⁻¹[1:m-1, 1:m-1] = G - g * g' / γ # inv(A'A)
    return x, P
end

################################# LACE #########################################
# Least Absolute Coefficient Elimination
struct LACE{T, AT<:AbstractMatOrFac{T}, BT<:AbstractVector{T}, V<:AbstractVector{T}, FT}
    A::AT # dictionary
    b::BT # target
    r::V # residual
    AiQR::FT # UpdatableQR
end
function LACE(A::AbstractMatOrFac, b::AbstractVector)
    r = similar(b)
    n, m = size(A)
    n ≥ m || throw("A needs to be overdetermined but is of size ($n, $m)")
    AiQR = UpdatableQR(A)
    LACE(A, b, r, AiQR)
end

function lace(A::AbstractMatOrFac, b::AbstractVector;
        max_residual::Real = Inf, max_increase::Real = Inf, sparsity::Int = 0)
    return lace(A, b, max_residual, max_increase, sparsity)
end

# A is overdetermined linear system, b is target, ε is tolerable residual norm
function lace(A::AbstractMatOrFac, b::AbstractVector, ε::Real, δ::Real, k::Int)
    n, m = size(A)
    L = LACE(A, b)
    x = sparse(ones(eltype(A), size(A, 2)))
    ldiv!!(x.nzval, L.AiQR, L.b, L.r)
    for _ in m:-1:k+1
        backward_step!(L, x, ε, δ) || break
    end
    return x
end
# input x is assumed to be ls-solution with current active set
function update!(P::LACE, x::SparseVector)
    i = argmin(abs, x.nzval) # choose least absolute coefficient magnitude from current support
    _dropindex!(x, P.AiQR, i) # drops ith atom in active set
    ldiv!!(x.nzval, P.AiQR, P.b, P.r)
    return x
end

function backward_step!(P::LACE, x::SparseVector, max_ε::Real, max_δ::Real)
    nnz(x) > 0 || return false
    residual!(P.r, P.A, x, P.b)
    normr = norm(P.r)

    i = argmin(abs, x.nzval) # choose least absolute coefficient magnitude from current support
    j = x.nzind[i] # remember which atom we are deleting
    _dropindex!(x, P.AiQR, i) # i is index into x.nzval, NOT into x
    ldiv!!(x.nzval, P.AiQR, P.b, P.r)

    residual!(P.r, P.A, x, P.b) # new residual
    δ² = norm(P.r)^2 - normr^2 # change in residual magnitude
    newnorm = sqrt(normr^2 + δ²)
    if newnorm < max_ε && δ² < max_δ^2
        return true
    else
        Aj = @view P.A[:,j] # if we can't accept the deletion,
        addindex!(x, P.AiQR, Aj, j) # add back the atom we deleted
        ldiv!!(x.nzval, P.AiQR, P.b, P.r)
        return false
    end
end
