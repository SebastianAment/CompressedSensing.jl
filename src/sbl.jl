########################## Sparse Bayesian Learning ############################
# sparse bayesian learning for linear regression
# IDEA: unification of all sbl algorithms via functional α_update
struct SparseBayesianLearning{T, AT, BT, GT} <: Update{T}
    AΣA::AT
    AΣb::BT
    Γ::GT
    # B # pre-allocate
    # Σ::ST
    function SparseBayesianLearning(A::AbstractMatrix, b::AbstractVector,
                        Σ::AbstractMatrix, Γ::Diagonal = (1.0I)(size(A, 2)))
        AΣ = A'inverse(Σ)
        AΣb = AΣ * b
        AΣA = AΣ * A
        T = eltype(A)
        new{T, typeof(AΣA), typeof(AΣb), typeof(Γ)}(AΣA, AΣb, Γ)
    end
end
const SBL = SparseBayesianLearning
# IDEA: prune indices? with SparseArrays
# for better conditioning
# C = (get_sqrt(Γ)*AΣA*get_sqrt(Γ) + I)
# B = inverse(get_sqrt(Γ)) * C * inverse(get_sqrt(Γ))
# CG could work well, because we have good initial x's between iterations
# for noiseless L0 norm minimization: B = get_sqrt(Γ)*pseudoinverse(A * get_sqrt(Γ))
function update!(S::SBL, x::AbstractVector = zeros(size(S.Γ, 2)))
    AΣA, AΣb, Γ = S.AΣA, S.AΣb, S.Γ
    B = AΣA + inv(Γ) # woodury makes sense if number of basis functions is larger than x
    B = factorize(B) # could use in-place cholesky!(B)
    B isa Number ? (x .= AΣb ./ B) : ldiv!(x, B, AΣb) # x = B \ AΣb, could generalize with functional solve! input
    # update rules
    # @. Γ.diag = x^2 + $diag($inverse(B)) + 1e-16 # provably convergent
    @. Γ.diag = x^2 / (1 - $diag($inverse(B)) / Γ.diag) + 1e-14 # good heuristic
    return x
end

sbl(A, b, σ²::Real) = sbl(A, b, σ²*I(length(b)))
function sbl(A::AbstractMatrix, b::AbstractVector, Σ::AbstractMatrix;
                        maxiter::Int = 128size(A, 2), min_change::Real = 1e-6)
    P = SBL(A, b, Σ)
    x = zeros(size(A, 2))
    γ_old = copy(P.Γ.diag)
    for i in 1:maxiter
        update!(P, x)
        if norm(γ_old - P.Γ.diag) < min_change
            break
        end
        γ_old .= P.Γ.diag
    end
    return x
end

# updating noise variance (if scalar)
# B⁻¹ = inverse(B)
# σ² = sum(abs2, b-A*x) / sum(i->B[i]/Γ[i], diagind(Γ)) # could replace with euclidean norm

##################### greedy marginal likelihood optimization ##################
# updates the prior variance which leads to the largest
# increase in the marginal likelihood
struct FastSparseBayesianLearning{T, AT<:AbstractMatrix{T}, BT, NT, A, CT} <: Update{T}
    A::AT
    b::BT
    Σ::NT
    α::A

    # temporaries
    S::A
    Q::A
    δ::A # holds difference in marginal likelihood for updating an atom
    C⁻¹::CT # TODO: use Woodbury type here
end
const FSBL = FastSparseBayesianLearning
function FSBL(A::AbstractMatrix, b::AbstractVector, Σ)
    α = fill(Inf, size(A, 2))
    δ = similar(α)
    Σ = factorize(Σ)
    ΣA = Σ \ A
    Q = ΣA' * b
    ΣA .*= A
    S = vec(sum(ΣA, dims = 1))
    C⁻¹ = Matrix(inv(Σ))
    FSBL(A, b, Σ, α, S, Q, δ, C⁻¹)
end

function FSBL(A::AbstractMatrix, b::AbstractVector, σ²::Real)
    FSBL(A, b, σ²*I(length(b)))
end

##################### Relevance Matching Pursuit (RMP_σ) #########################
# IDEA: could merge FSBL and RMPS types
struct RMPS{T, AT<:AbstractMatrix{T}, BT, NT, A, CT} <: Update{T}
    A::AT
    b::BT
    Σ::NT
    α::A

    # temporaries
    S::A
    Q::A
    δ::A # holds difference in marginal likelihood for updating an atom
    C⁻¹::CT # IDEA: use Woodbury type here
end
function RMPS(A::AbstractMatrix, b::AbstractVector, σ²::Real, α::AbstractVector = fill(Inf, size(A, 2)))
    RMPS(A, b, σ²*I(length(b)), α)
end

# α is the inverse of γ, the prior variances of x
function RMPS(A::AbstractMatrix, b::AbstractVector, Σ::AbstractMatOrFac,
                                    α::AbstractVector = fill(Inf, size(A, 2)))
    n, m = size(A)
    δ = similar(α)
    C = if all(isinf, α)
            Σ
        else
            i = .!isinf.(α)
            γ = inv.(α[i])
            Γ = Diagonal(γ)
            Ai = A[:, i]
            Woodbury(Σ, Ai, Γ, Ai')
        end
    C = factorize(C)
    CA = C \ A
    Q = CA' * b
    CA .*= A
    S = vec(sum(CA, dims = 1))
    C⁻¹ = Matrix(inv(C))
    RMPS(A, b, Σ, α, S, Q, δ, C⁻¹)
end

# TODO: this can be done more efficiently with P.C
function Base.getproperty(S::Union{FSBL, RMPS}, s::Symbol)
    if s == :x
        active = @. !isinf(S.α) # active basis patterns
        x = spzeros(eltype(S.A), size(S.A, 2))
        Ai = @view S.A[:, active]
        Σ⁻¹ = inverse(S.Σ)
        P = inverse(*(Ai', Σ⁻¹, Ai) + Diagonal(S.α[active]))
        x[active] .= P * (Ai' * (Σ⁻¹ * S.b))
        dropzeros!(x)
    else
        getfield(S, s)
    end
end

################################################################################
# optimizes prior variances, returns corresponding weight estimate
# min_increase is the minimum increase in the marginal likelihood below which
# the algorithm terminates
function fsbl(A, b, Σ; maxiter = 2size(A, 2), min_increase = 1e-6)
    P = FSBL(A, b, Σ)
    optimize!(P, maxiter = maxiter, min_increase = min_increase)
    P.x
end
# optimizes prior variances, returns FSBL object
function optimize!(P::FSBL; maxiter = 2size(A, 2), min_increase = 1e-2)
    for i in 1:maxiter
        update!(P, P.α)
        if maximum(P.δ) < min_increase
            break
        end
    end
    return P
end

function update!(P::FSBL, α::AbstractVector = P.α)
    i = sbl_index!(P) # index corresponding to highest increase in marginal likelihood
    active, relevant = isactive(P, i), isrelevant(P, i)
    if !active && relevant
        sbl_acquisition!(P, i)
    elseif active && !relevant
        sbl_deletion!(P, i)
    elseif active && relevant
        sbl_update!(P, i)
    end # else, no action required
    return α
end

# updates set of active basis functions
# greedy strategy: chooses update which increases marginal likelihood most (max δ)
function update_α!(α, δ, S, Q)
    k = argmax(δ)
    sk, qk = get_sq(S[k], Q[k], α[k])
    α[k] = optimal_α(sk, qk)
    return α
end

############################## basic helpers ###################################
isactive(P::FSBL, i::Int) = (P.α[i] < Inf)
function isrelevant(P::FSBL, i::Int)
    s, q = get_sq(P.S[i], P.Q[i], P.α[i])
    return s < q^2
end
# calculate small s, q from S, Q (see Tipping 2003)
# also known as sparsity and quality factors
function get_sq(S::Real, Q::Real, α::Real)
    (S, Q) .* (α < Inf ? α / (α-S) : 1)
end

function optimal_α(s::Real, q::Real)
    (s < q^2) ? s^2 / (q^2 - s) : Inf
end

function sbl_index!(P::FSBL)
    @. P.δ = delta(P.α, P.S, P.Q)
    i = argmax(P.δ)
end

# potential change in marginal likelihood, and best action
function delta(α::Real, S::Real, Q::Real)
   s, q = get_sq(S, Q, α)
   active = α < Inf
   relevant = s < q^2
   if !active && relevant # out of model and high quality factor
       return δ_add(S, Q)
   elseif active && !relevant # in model but not high enough quality
       return δ_delete(S, Q, α)
   elseif active && relevant # α re-estimation
       αn = optimal_α(s, q)
       return δ = δ_update(S, Q, α, αn)
   else # !active && !relevant
       return 0.
   end
end

############################# acquisition ######################################
function sbl_acquisition!(P::Union{FSBL, RMPS})
    @. P.δ = sbl_acquisition_value(P.α, P.S, P.Q)
    k = argmax(P.δ) # choosing maximum q/s value, corresponds to maximum increase in likelihood
    return sbl_acquisition!(P, k)
end
function sbl_acquisition!(P::Union{FSBL, RMPS}, i::Int)
    if P.δ[i] > 0 # only add if index i is beneficial
        si, qi = get_sq(P.S[i], P.Q[i], P.α[i]) # updating α
        P.α[i] = optimal_α(si, qi)
        γ = inv(P.α[i])
        update_SQC!(P, i, γ)
        return true
    end
    return false
end
function sbl_acquisition_value(α::Real, S::Real, Q::Real)
    s, q = get_sq(S, Q, α)
    active = α < Inf
    relevant = s < q^2
    return (!active && relevant) ? δ_add(S, Q) : 0.
end

function δ_add(S::Real, Q::Real)
   (Q^2 - S) / S + log(S) - log(Q^2)
end

function rmp_acquisition_value(α::Real, S::Real, Q::Real)
    s, q = get_sq(S, Q, α)
    active = α < Inf
    relevant = s < q^2
    if !isactive && isrelevant
        q^2 / s
    else
        0.
    end
end

################################ update ########################################
# delete index with maximum increase in likelihood
function sbl_update!(P::Union{FSBL, RMPS})
    @. P.δ = sbl_update_value(P.α, P.S, P.Q) # only necessary for active atoms
    i = argmax(P.δ)
    return sbl_update!(P, i)
end
# delete index i, if it increases the likelihood
function sbl_update!(P::Union{FSBL, RMPS}, i::Int)
    if P.δ[i] > 0
        si, qi = get_sq(P.S[i], P.Q[i], P.α[i])
        αs = optimal_α(si, qi)
        γ = inv(αs) - inv(P.α[i]) # difference between new γ and old γ
        update_SQC!(P, i, γ)
        P.α[i] = αs
        return P.δ[i]
    end
    return zero(eltype(P.δ[i]))
end

function sbl_update_value(α::Real, S::Real, Q::Real)
    α < Inf || return zero(α) # is active
    s, q = get_sq(S, Q, α)
    s < q^2 || return zero(α) # is relevant
    αn = optimal_α(s, q)
    return δ_update(S, Q, α, αn)
end
# potential change in marginal likelihood upon updating α to αn
function δ_update(S::Real, Q::Real, α::Real, αn::Real)
   d = 1/αn - 1/α
   Q^2 / (S + 1/d) - log(max(1 + S*d, 0))
end

################################ deletion ######################################
function sbl_deletion!(P::Union{FSBL, RMPS})
    @. P.δ = sbl_deletion_value(P.α, P.S, P.Q)
    i = argmax(P.δ)
    return sbl_deletion!(P, i)
end
function sbl_deletion!(P::Union{FSBL, RMPS}, i::Int)
    if P.δ[i] > 0
        si, qi = get_sq(P.S[i], P.Q[i], P.α[i]) # updating α
        γ = -inv(P.α[i]) # difference between new γ (0) and old γ (1/α[k])
        P.α[i] = optimal_α(si, qi) # setting to Inf
        update_SQC!(P, i, γ)
        return true
    end
    return false
end
function sbl_deletion_value(α::Real, S::Real, Q::Real)
    α < Inf || return zero(α) # is active
    s, q = get_sq(S, Q, α)
    s > q^2 || return zero(α) # is irrelevant
    return δ_delete(S, Q, α)
end
# potential change in marginal likelihood upon setting α to Inf
function δ_delete(S::Real, Q::Real, α::Real)
   Q^2 / (S - α) - log(1 - (S/α)) # delete k
end

######################## computing and updating S, Q, C⁻¹ ######################
get_SQ!(P::RMPS) = get_SQ!(P.S, P.Q, P.A, P.b, P.C⁻¹)
function get_SQ!(S, Q, A, b, C⁻¹) # try to avoid
    for k in 1:size(A, 2)
        Ak = @view A[:, k]
        S[k] = dot(Ak, C⁻¹, Ak)
        Q[k] = dot(Ak, C⁻¹, b)
    end
    S, Q
end

# get kernel matrix C = Σ + A * Γ * A'
function get_C_inverse(isactive, A, α, Σ)
    if any(isactive)
        Ai = @view A[:, isactive]
        Γ = Diagonal(inv.(α[isactive]))
        C = Woodbury(Σ, Ai, Γ, Ai')
        C⁻¹ = inverse(factorize(C))
    else
        C⁻¹ = inverse(Σ)
    end
end

# update after rank one modification
function update_C_inverse!(P::Union{FSBL, RMPS}, i::Int, α::Real)
    a = @view P.A[:, i]
    v = P.C⁻¹ * a
    @. P.C⁻¹ -= v * v' / (α + P.S[i]) # update
end

# updates C⁻¹, S, Q in P corresponding to rank one correction of C
# with ith atom with coefficient γ: C + γ * ai * ai'
function update_SQC!(P::Union{FSBL, RMPS}, i::Int, γ::Real)
    α = inv(γ)
    a = P.A[:, i]
    v = P.C⁻¹ * a
    update_C_inverse!(P, i, α)
    Av = P.A' * v
    si, qi = P.S[i], P.Q[i]
    @. P.S -= Av^2 / (α + si)
    @. P.Q -= Av * qi / (α + si)
    return P.S, P.Q
end

################################################################################
# Σ is the noise variance (scalar or matrix)
# maxiter is the maximum number of outer iterations
# maxiter_acquisition is the maximum number of iterations per aquisition stage
# maxiter_deletion is the maximum number of iterations per deletion stage
# min_increase is the minimum increase in marginal likelihood, below which the algorithm terminates
function rmps(A, b, Σ; maxiter::Int = size(A, 1), maxiter_acquisition::Int = size(A, 1),
                    maxiter_deletion::Int = size(A, 1), min_increase::Real = 1e-6) # in RMP experiments this was min_increase = 1e-2
    P = RMPS(A, b, Σ)
    optimize!(P, maxiter = maxiter, maxiter_acquisition = maxiter_acquisition,
                maxiter_deletion = maxiter_deletion, min_increase = min_increase)
    return P.x
end

function optimize!(P::RMPS; maxiter::Int = size(P.A, 1),
                            maxiter_acquisition::Int = size(P.A, 1),
                            maxiter_deletion::Int = size(P.A, 1),
                            min_increase::Real = 1e-6)
    A, b, Σ, S, Q = P.A, P.b, P.Σ, P.S, P.Q
    α = P.α
    α .= Inf
    old_α = copy(α)
    for i in 1:maxiter
        for _ in 1:maxiter_acquisition # acquisition stage
            rmp_acquisition!(P) || break # if we did not add any atom, we're done
        end
        old_α != α || break # while norm of α is still changing
        old_α .= α
        for j in 1:maxiter_deletion # update and deletion stage
            if rmp_deletion!(P)
                continue
            elseif rmp_update!(P) < min_increase
                break
            end
        end
        old_α != α || break # while norm of α is still changing
        old_α .= α
    end
    return P
end

# acquisition and update are identical to fast sbl
rmp_acquisition!(P::RMPS) = sbl_acquisition!(P)
rmp_update!(P::RMPS) = sbl_update!(P)

################################ deletion ######################################
function rmp_deletion!(P::RMPS)
    α, S, Q = P.α, P.S, P.Q
    @. P.δ = rmp_deletion_value(α, S, Q)
    min_δ, k = findmin(P.δ) # choosing minimum q/s value
    if min_δ < 1 # only delete if we are not doing damage
        sk, qk = get_sq(S[k], Q[k], α[k]) # updating α
        γ = -inv(α[k]) # difference between new γ (0) and old γ (1/α[k])
        α[k] = optimal_α(sk, qk) # setting to Inf
        update_SQC!(P, k, γ)
        return true
    end
    return false
end

# WARNING: lower is better here
function rmp_deletion_value(α::Real, S::Real, Q::Real)
    s, q = get_sq(S, Q, α)
    isactive = α < Inf
    isrelevant = s < q^2
    if isactive && !isrelevant # in model but not high enough quality
        δ = q^2 / s
    else
        Inf
    end
end

##################### optimization of noise variance ###########################
# optimizes noise variance in an outer loop above rmps
# a_σ², b_σ² are shape and scale parameter for Inverse Gamma prior on σ²
function rmps(A::AbstractMatrix, b::AbstractVector, optimize_σ²::Val{true},
                        σ²::Real = 1e-2, a_σ²::Real = 0, b_σ²::Real = 0;
                        maxiter::Int = 2size(A, 2), min_increase::Real = 1e-6,
                        maxouteriter::Int = 16, min_change::Real = 1e-12)
    α = fill(Inf, size(A, 2))
    for i in 1:maxouteriter
        P = RMPS(A, b, σ², α)
        optimize!(P, maxiter = maxiter, min_increase = min_increase)
        α = P.α
        σ²_new = estimate_σ²(P, a_σ², b_σ²)
        converged = abs(σ²_new - σ²) < min_change
        σ² = σ²_new
        if converged
            break
        end
    end
    x = RMPS(A, b, σ², α).x
    return x, σ²
end

function estimate_σ²(P::RMPS, a_σ²::Real = 0, b_σ²::Real = 0)
    estimate_σ²(P.A, P.x, P.b, inv.(P.α), a_σ², b_σ²)
end
# with inverse-gamma prior on σ² with shape and scale parameters a, b
function estimate_σ²(A::AbstractMatrix, x::AbstractVector, b::AbstractVector,
                    γ::AbstractVector, a_σ²::Real = 0, b_σ²::Real = 0)
    n = size(A, 1)
    (sum(abs2, b-A*x) + 2b_σ²) / (n - sum(γ) + 2a_σ²)
end
