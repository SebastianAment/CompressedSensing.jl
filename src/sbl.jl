########################## Sparse Bayesian Learning ############################
# sparse bayesian learning for linear regression
# TODO: investigate direct optimization of nlml, not via em
# ... and unification of all sbl algorithms via functional α_update
struct SparseBayesianLearning{T, AT, BT, GT} <: Update{T}
    AΣA::AT
    AΣb::BT
    Γ::GT
    # B # pre-allocate
    # Σ::ST
    function SparseBayesianLearning(A::AbstractMatrix, b::AbstractVector,
                        Σ::AbstractMatOrUni, Γ::Diagonal = (1.0I)(size(A, 2)))
        AΣ = A'inverse(Σ)
        AΣb = AΣ * b
        AΣA = AΣ * A
        T = eltype(A)
        new{T, typeof(AΣA), typeof(AΣb), typeof(Γ)}(AΣA, AΣb, Γ)
    end
end
const SBL = SparseBayesianLearning
# TODO: prune indices? with SparseArrays
# for better conditioning
# C = (sqrt(Γ)*AΣA*sqrt(Γ) + I)
# B = inverse(sqrt(Γ)) * C * inverse(sqrt(Γ))
# CG could work well, because we have good initial x's between iterations
# for noiseless L0 norm minimization: B = sqrt(Γ)*pseudoinverse(A * sqrt(Γ))
function update!(S::SBL, x::AbstractVector = zeros(size(S.Γ, 2)))
    AΣA, AΣb, Γ = S.AΣA, S.AΣb, S.Γ
    B = AΣA + inverse(Γ) # woodury makes sense if number of basis functions is larger than x
    B = factorize(B) # could use in-place cholesky!(B)
    B isa Number ? (x .= AΣb ./ B) : ldiv!(x, B, AΣb) # x = B \ AΣb, could generalize with functional solve! input
    # update rules
    # @. Γ.diag = x^2 + $diag($inverse(B)) + 1e-16 # provably convergent
    @. Γ.diag = x^2 / (1 - $diag($inverse(B)) / Γ.diag) + 1e-14 # good heuristic
    return x
end

sbl(A, b, σ::Real) = sbl(A, b, σ^2*I(length(b)))
function sbl(A::AbstractMatrix, b::AbstractVector, Σ::AbstractMatOrUni;
                        maxiter::Int = size(A, 1), min_change::Real = 1e-6)
    P = SBL(A, b, Σ)
    x = zeros(size(A, 2))
    for _ in 1:maxiter
        update!(P, x)
    end
    return x
end

# updating noise variance (if scalar)
# B⁻¹ = inverse(B)
# σ² = sum(abs2, b-A*x) / sum(i->B[i]/Γ[i], diagind(Γ)) # could replace with euclidean norm

##################### greedy marginal likelihood optimization ##################
# updates the prior variance which leads to the largest
# increase in the marginal likelihood
struct GreedySparseBayesianLearning{T, AT<:AbstractMatrix{T}, BT, NT, A} <: Update{T}
    A::AT
    b::BT
    Σ::NT
    α::A

    # temporaries
    S::A
    Q::A
    δ::A # holds difference in marginal likelihood for updating an atom
end
const GSBL = GreedySparseBayesianLearning
function GSBL(A::AbstractMatrix, b::AbstractVector, Σ)
    α = fill(Inf, size(A, 2))
    S, Q, δ = (similar(α) for i in 1:3)
    GSBL(A, b, Σ, α, S, Q, δ)
end

function GSBL(A::AbstractMatrix, b::AbstractVector, σ::Real)
    GSBL(A, b, σ^2*I(length(b)))
end

function optimize!(P::GSBL; maxiter = 128, dl = 1e-2)
    for i in 1:maxiter
        update!(P, P.α)
        if maximum(P.δ) < dl
            break
        end
    end
    return P
end

function greedy_sbl(A, b, σ; maxiter = 256, dl = 1e-4)
    P = GSBL(A, b, σ)
    optimize!(P, maxiter = maxiter, dl = dl)
    P.x
end

function update!(P::GSBL, α::AbstractVector)
    isactive = @. !isinf(α) # active basis patterns
    A, b, Σ, S, Q = P.A, P.b, P.Σ, P.S, P.Q
    # get kernel matrix C
    if any(isactive)
        Ai = @view A[:, isactive]
        Γ = Diagonal(inv.(α[isactive]))
        C = Woodbury(Σ, Ai, Γ, Ai')
        C⁻¹ = inverse(factorize(C))
    else
        C⁻¹ = inverse(Σ)
    end
    # update sparsity and quality factors
    @threads for k in 1:size(A, 2)
        Ak = @view A[:, k]
        S[k] = sparsity(C⁻¹, Ak)
        Q[k] = quality(C⁻¹, b, Ak)
    end
    # potential change in marginal likelihood for each atom
    @. P.δ = delta(α, S, Q)
    return update_α!(α, P.δ, S, Q)
end

function sparsity(C⁻¹::AbstractMatOrFac, a::AbstractVector)
    dot(a, C⁻¹, a)
end
function quality(C⁻¹::AbstractMatOrFac, t::AbstractVector, a::AbstractVector)
    dot(a, C⁻¹, t)
end

# updates set of active basis functions
# greedy strategy: chooses update which increases marginal likelihood most (max δ)
function update_α!(α, δ, S, Q)
    k = argmax(δ)
    sk, qk = sq(S[k], Q[k], α[k])
    α[k] = optimal_α(sk, qk)
    return α
end

# potential change in marginal likelihood for each atom
function delta(α::Real, S::Real, Q::Real)
    s, q = sq(S, Q, α)
    isactive = α < Inf
    isrelevant = s < q^2
    if !isactive && isrelevant # out of model and high quality factor
        δ = δ_add(S, Q)
    elseif isactive && !isrelevant # in model but not high enough quality
        δ = δ_delete(S, Q, α)
    elseif isactive && isrelevant # α re-estimation
        αn = optimal_α(s, q)
        δ = δ_update(S, Q, α, αn)
    else # !isactive && !isrelevant
        δ = 0.
    end
    return δ
end

function δ_add(S, Q)
    (Q^2 - S) / S + log(S) - log(Q^2) # add k
end
function δ_delete(S, Q, α)
    Q^2 / (S - α) - log(1 - (S/α)) # delete k
end
function δ_update(S, Q, α, αn)
    d = 1/αn - 1/α
    Q^2 / (S + 1/d) - log(max(1 + S*d, 0))
end

# calculate small s, q from S, Q (see Tipping 2003)
# also known as sparsity and quality factors
function sq(S::Real, Q::Real, α::Real)
    if α < Inf
        s = α*S / (α-S)
        q = α*Q / (α-S)
        s, q
    else
        S, Q
    end
end

function optimal_α(s::Real, q::Real)
    s < q^2 ? s^2 / (q^2 - s) : Inf
end

##################### Relevance Matching Pursuit (RMP_σ) #########################
struct RMPS{T, AT<:AbstractMatrix{T}, BT, NT, A, CT} <: Update{T}
    A::AT
    b::BT
    Σ::NT
    α::A

    # temporaries
    S::A
    Q::A
    δ::A # holds difference in marginal likelihood for updating an atom
    C⁻¹::CT
end
function RMPS(A::AbstractMatrix, b::AbstractVector, σ::Real)
    RMPS(A, b, σ^2*I(length(b)))
end
function RMPS(A::AbstractMatrix, b::AbstractVector, Σ::AbstractMatOrFac)
    n, m = size(A)
    α = fill(Inf, m)
    S, Q, δ = (similar(α) for i in 1:3)
    isactive = @. !isinf(α)
    C⁻¹ = get_C_inverse(isactive, A, α, Σ)
    C⁻¹ = Matrix(C⁻¹)
    RMPS(A, b, Σ, α, S, Q, δ, C⁻¹)
end
function Base.getproperty(S::Union{GSBL, RMPS}, s::Symbol)
    if s == :x
        isactive = @. !isinf(S.α) # active basis patterns
        x = spzeros(eltype(S.A), size(S.A, 2))
        Ai = @view S.A[:, isactive]
        Σ⁻¹ = inverse(S.Σ)
        P = inverse(*(Ai', Σ⁻¹, Ai) + Diagonal(S.α[isactive]))
        x[isactive] .= P * (Ai' * (Σ⁻¹ * S.b))
        dropzeros!(x)
    else
        getfield(S, s)
    end
end

# σ is the noise variance
# maxiter is the maximum number of outer iterations
# maxiter_acquisition is the maximum number of iterations per aquisition stage
# maxiter_deletion is the maximum number of iterations per deletion stage
# dl is the minimum increase in marginal likelihood, below which the algorithm terminates
function rmps(A, b, σ; maxiter::Int = 2,
                    maxiter_acquisition::Int = size(A, 1),
                    maxiter_deletion::Int = size(A, 1), dl = 1e-2) # in RMP experiments this was dl = 1e-2
    P = RMPS(A, b, σ)
    A, b, Σ, S, Q = P.A, P.b, P.Σ, P.S, P.Q
    α = P.α
    α .= Inf
    for i in 1:maxiter # while norm of α is still changing

        for _ in 1:maxiter_acquisition # acquisition stage
            isactive = @. !isinf(α) # active basis patterns
            C⁻¹ = get_C_inverse(isactive, A, α, Σ) # update C, sparsity and quality factors
            P.C⁻¹ .= Matrix(C⁻¹)
            println("in iter")
            display(P.C⁻¹)
            println(S)
            update_SQ!(S, Q, A, b, C⁻¹)
            println(S)
            rmp_acquisition!(P) || break # if we did not add any atom, we're done
            # println(findall(!=(Inf), α))
        end

        for j in 1:maxiter_deletion # update and deletion stage
            isactive = @. !isinf(α) # active basis patterns
            C⁻¹ = get_C_inverse(isactive, A, α, Σ)
            update_SQ!(S, Q, A, b, C⁻¹)
            if rmp_deletion!(α, S, Q)
                continue
            elseif rmp_update!(α, S, Q) < dl
                break
            end
        end

    end
    return P.x
end

function update_SQ!(S, Q, A, b, C⁻¹)
    @threads for k in 1:size(A, 2) # parallelized
        Ak = @view A[:, k]
        S[k] = sparsity(C⁻¹, Ak)
        Q[k] = quality(C⁻¹, b, Ak)
    end
    S, Q
end

function get_C_inverse(isactive, A, α, Σ)
    # get kernel matrix C
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
function update_C_inverse!(P::RMPS, i::Int, α::Real)
    a = @view P.A[:, i]
    v = P.C⁻¹ * a
    @. P.C⁻¹ -= v * v' / (α + P.S[i]) # update
end

############################# acquisition ######################################
function rmp_acquisition!(P::RMPS)
    A, α, S, Q = P.A, P.α, P.S, P.Q
    δ = @. rmp_acquisition_value(α, S, Q)
    k = argmax(δ)
    if δ[k] > 0 # only add if it is beneficial
        sk, qk = sq(S[k], Q[k], α[k]) # updating α
        α[k] = optimal_α(sk, qk)
        add_update!(P, k)
        # update_C_inverse!(P, k, α[k])
        return true
    end
    return false
end

function add_update!(P::RMPS, i::Int)
    a = P.A[:, i]
    v = P.C⁻¹ * a
    println("in update")
    println(P.α[i])
    # println(P.S)
    P.S .-= (P.A * v).^2 ./ (P.α[i] + P.S[i])
    # println(P.S)
    display(P.C⁻¹)
    update_C_inverse!(P, i, P.α[i])
    display(P.C⁻¹)
    return P.S, P.Q
end

function rmp_acquisition_value(α::Real, S::Real, Q::Real)
    s, q = sq(S, Q, α)
    isactive = α < Inf
    isrelevant = s < q^2
    if !isactive && isrelevant
        q^2 / s
    else
        0.
    end
end


################################ update ########################################
function rmp_update!(α::AbstractVector, S::AbstractVector, Q::AbstractVector)
    δ = @. rmp_update_value(α, S, Q)
    k = argmax(δ)
    if δ[k] > 0 # only update if it is beneficial
        sk, qk = sq(S[k], Q[k], α[k]) # updating α
        α[k] = optimal_α(sk, qk)
        return δ[k]
    end
    return zero(eltype(δ))
end

function rmp_update_value(α::Real, S::Real, Q::Real)
    s, q = sq(S, Q, α)
    isactive = α < Inf
    isrelevant = s < q^2
    if isactive && isrelevant # α re-estimation
        αn = optimal_α(s, q)
        δ_update(S, Q, α, αn)
    else
        zero(α)
    end
end

################################ deletion ######################################
function rmp_deletion!(α::AbstractVector, S::AbstractVector, Q::AbstractVector)
    δ = @. rmp_deletion_value(α, S, Q)
    k = argmin(δ) # choosing minimum q/s value
    if δ[k] < 1
        sk, qk = sq(S[k], Q[k], α[k]) # updating α
        α[k] = optimal_α(sk, qk) # setting to Inf
        return true
    end
    return false
end

# WARNING: lower is better here
function rmp_deletion_value(α::Real, S::Real, Q::Real)
    s, q = sq(S, Q, α)
    isactive = α < Inf
    isrelevant = s < q^2
    if isactive && !isrelevant # in model but not high enough quality
        δ = q^2 / s
    else
        Inf
    end
end


# function update_SQ!(P::RMPS, i::Int)
#     P.S .-= P.S -
# end


# acquires new atom
# TODO: update S, Q accordingly
# updates S, Q after adding i and setting variance to optimal value
# function add_update!(P::RMPS, i::Int)
#     # β = 1/P.Σ.λ # 1/σ^2
#     α, S, Q, A = P.α, P.S, P.Q, P.A
#     β = 1/1e-4
#     Σ = inv(Diagonal(α) + β * A'A)
#     s_ii = (α[i] .+ S[i]).^(-1)
#     m_i = s_ii * Q[i]
#     a = @view A[:, i]
#     ei = a .- β * (A * (Σ * (A' * a)))
#     # temp = Σ*A'*a
#     mCi = β * (A' * ei)
#     @. S -= s_ii * mCi^2
#     @. Q -= m_i * mCi
#     # for m in 1:length(S)
#         # mCi = (β * dot(A[:, m], ei))
#         # S[m] -= s_ii * mCi^2
#         # Q[m] -= m_i * mCi
#     # end
#     return S, Q
# end
