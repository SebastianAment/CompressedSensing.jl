########################## Bayesian OMP ########################################
# WARNING: it's an untested sketch
# in their experiments, does not notably improve over standard OMP
# "Bayesian Pursuit Algorithms"
# struct BayesianOrthogonalMatchingPursuit{T, AT<:AbstractMatrix{T},
#                                     B<:AbstractVector{T}, P,
#                                     ST<:AbstractVector{<:Bool}} <: AbstractMatchingPursuit{T}
#     A::AT
#     b::B
#     k::Int # maximum number of non-zeros
#     #
#     σ::T # noise standard deviation
#
#     σₓ::T # Normal prior over x
#     p::P # Bernoulli prior over s
#
#     # temporary storage
#     r::B # residual
#     Ar::B # inner products between measurement matrix and residual
#     Ai::AT # space for A[:, x.nzind] and its qr factorization
# end
# const BOMP = BayesianOrthogonalMatchingPursuit
# function BOMP(A::AbstractMatrix, b::AbstractVector, k::Integer)
#     n, m = size(A)
#     T = eltype(A)
#     r, Ar = zeros(T, n), zeros(T, m)
#     Ai = zeros(T, (n, k))
#     BOMP(A, b, k, r, Ar, Ai)
# end
#
# function update!(P::BOMP, x::AbstractVector = spzeros(size(P.A, 2)))
#     # nnz(x) ≥ P.k && return x # return if the maximum number of non-zeros was reached
#     residual!(P, x)
#     # T = 2σ^2 * (σₓ^2 + σ^2) / σₓ^2 * log((1-p)/p) # assuming every atom has the same Bernoulli prior
#     ε = σ^2 / σₓ^2
#     λ = σ^2 * log((1-p)/p)
#     T = 2λ * (1 + ε) # assuming every atom has the same Bernoulli prior
#     st = zeros(Bool, length(x)) # s tilde
#     xt = zero(x)
#     for i in eachindex(s)
#         di = @view P.A[:, i]
#         inner = dot(P.r, di) + x[i] * dot(di, di)
#         st[i] = inner^2 > T
#         if st[i]
#             xt[i] = σₓ^2 / (σₓ^2 + σ^2) * (x[i] + dot(P.r, di)) # x tilde
#         end
#     end
#     ρ = similar(x)
#     for i in eachindex(x)
#         ρ[i] = -sum(abs2, P.r + (x[i] - xt[i]) * di) - ε*xt[i]^2 - λ*st[i]
#     end
#     i = argmax(ρ)
#     x[i] = st[i] ? NaN : 0 # add non-zero index to x
#     dropzeros!(x)
#     Ai = P.A[:, x.nzind]
#     x.nzval .= (Ai'Ai + ε * I) \ Ai'P.b # optimize all active atoms
#     return x
# end
#
# # calculates k-sparse approximation to Ax = b via orthogonal matching pursuit
# function bomp(A::AbstractMatrix, b::AbstractVector, k::Int)
#     P! = BOMP(A, b, k)
#     x = spzeros(size(A, 2))
#     for i in 1:k
#         P!(x)
#     end
#     return x
# end

# abstract matching pursuit,
function amp!(P::AbstractMatchingPursuit,
            A::AbstractMatrix, b::AbstractVector, δ::Real, k::Int = size(A, 1),
                                                        x = spzeros(size(A, 2)))
    δ ≥ 0 || throw("δ = $δ has to be non-negative")
    for i in 1:k
        update!(P, x)
        norm(residual!(P, x)) ≥ δ || break
    end
    return x
end

# calculates energetic inner product of x, y w.r.t. I - ΦΦ⁺
# assuming Φ = QR, ΦΦ⁺ = Q*Q'
# TODO: pre-allocation?
# function LinearAlgebra.dot(x::AbstractVector, P::RMP, y::AbstractVector)
#     Q = P.AiQR.Q1
#     Qx = Q'x
#     Qy = x ≡ y ? Qx : Q'y
#     dot(x, y) - dot(Qx, Qy)
# end
#
# function LinearAlgebra.norm(P::RMP, φ::AbstractVector)
#     Q = P.AiQR.Q1
#     Qφ = Q'φ
#     sum(abs2, φ) - sum(abs2, Qφ)
# end
#
# function LinearAlgebra.norm(P::RMP, A::AbstractMatrix)
#     Q = P.AiQR.Q1
#     QA = Q'A
#     sum(abs2, A, dims = 1) - sum(abs2, QA, dims = 1)
# end


##################################### OLD #######################################
# old: in-loop deletion
# function rmp(A::AbstractMatrix, b::AbstractVector, δ::Real = 1e-12,
#             x = spzeros(size(A, 2)); maxiter = size(A, 1), rescale::Bool = false)
#     P = RMP(A, b, rescale = rescale)
#     for i in 1:maxiter
#         add_relevant!(P, x, δ) || break
#         for j in 1:maxiter # deletion stage
#             delete_redundant!(P, x, δ) || break
#         end
#     end
#     return x
# end

################################################################################
# function omp(A::AbstractMatrix, b::AbstractVector, k::Int)
#     P = OMP(A, b, k)
#     x = spzeros(size(A, 2))
#     for _ in 1:k
#         update!(P, x)
#     end
#     return x
# end

################################################################################
# struct OOMP{T, AT<:AbstractMatrix{T}, B<:AbstractVector{T}, FT} <: AbstractMatchingPursuit{T}
#     A::AT
#     b::B
#     # k::Int # maximum number of non-zeros
#
#     # temporary storage
#     r::B # residual
#     Ar::B # inner products between measurement matrix and residual
#     QA::AT # (k, m)
#     AiQR::FT # updatable QR factorization of Ai
#
#     rescaling::B # energetic renormalization
#     rescale::Bool # toggles rescaling by energetic norm
# end
# function update!(P::OOMP, x::AbstractVector)
#     residual!(P, x)
#
#     return -1
# end
# function oomp_index!() end
