function basispursuit(A::AbstractMatrix, b::AbstractVector)
    basispursuit(A, b, ones(eltype(A), size(A, 2)))
end
const bp = basispursuit

function basispursuit(A::AbstractMatrix, b::AbstractVector, w::AbstractVector)
    model = JuMP.Model(optimizer_with_attributes(Clp.Optimizer, "LogLevel" => 0))
    n, m = size(A)
    @variable(model, x⁺[1:m] ≥ 0)
    @variable(model, x⁻[1:m] ≥ 0)
    @objective(model, Min, dot(w, x⁺) + dot(w, x⁻))
    @constraint(model, constraint, A * (x⁺ .- x⁻) .== b)
    JuMP.optimize!(model)
    x = @. JuMP.value(x⁺) - JuMP.value.(x⁻)
    sparse(x)
end

function basispursuit_reweighting(A::AbstractMatrix, b::AbstractVector,
                                    reweighting!; maxiter = 8, min_decrease = 1e-8)
    x = basispursuit(A, b)
    w = ones(length(x))
    for i in 2:maxiter
        reweighting!(w, x) # updates w
        xs = basispursuit(A, b, w)
        if norm(xs-x) < min_decrease # stop if we don't make sufficient progress
            return xs
        end
        x .= xs
    end
    x
end

candes_weight(x, ε) = inv(abs(x) + ε)
function candes_weights!(w, x, ε::Real)
    @. w = candes_weight(x, ε)
    if any(isnan, w) || any(isinf, w)
        throw("weights contain NaN or Inf: $w")
    end
end
candes_function(ε::Real) = (w, x) -> candes_weights!(w, x, ε)
function candes_reweighting(A::AbstractMatrix, b::AbstractVector,
                                            ε = 1e-2; maxiter = 8)
    basispursuit_reweighting(A, b, candes_function(ε), maxiter = maxiter)
end
const bp_candes = candes_reweighting

# using LinearAlgebraExtensions: LowRank
# TODO: stop when w has converged, instead of maxiter
function ard_weights!(w, A, x, ε::Real, iter::Int = 8)
    if any(==(0), w)
        error("weights cannot be zero")
    end
    # nzind = x.nzind # this could speed up the weight computation
    # Ai = @view A[:, nzind]
    # wx = @. abs(x.nzval) / w[nzind]
    # K = Woodbury(ε*I(size(A, 1)), Ai, WX, Ai')
    for i in 1:iter
        wx = @. abs(x) / w
        WX = Diagonal(wx)
        K = Woodbury(ε*I(size(A, 1)), A, WX, A')
        K⁻¹ = inverse(factorize(K))
        dK(a) = sqrt(max(dot(a, K⁻¹, a), 0))
        @. w = dK($eachcol(A))
    end
end

ard_function(A::AbstractMatrix, ε::Real) = (w, x) -> ard_weights!(w, A, x, ε)

function ard_reweighting(A::AbstractMatrix, b::AbstractVector, ε::Real = 1e-2;
                                                            maxiter::Int = 8)
    basispursuit_reweighting(A, b, ard_function(A, ε), maxiter = maxiter)
end
const bp_ard = ard_reweighting

############################# Basis Pursuit Denoising ##########################
# noisy variant, where we allow δ-deviations from data
# minimizes l2 norm constrained linear objective (conic program)
# min_x |x|_1 s.t. |Ax-b|_2 < δ
function basis_pursuit_denoising(A::AbstractMatrix, b::AbstractVector, δ::Real,
                                    w::AbstractVector = ones(eltype(A), size(A, 2)))

    model = bpd_ecos_model()
    n, m = size(A)
    @variable(model, x⁺[1:m] ≥ 0)
    @variable(model, x⁻[1:m] ≥ 0)
    @objective(model, Min, dot(w, x⁺) + dot(w, x⁻))
    @constraint(model, [δ; A * (x⁺ .- x⁻) .- b] in SecondOrderCone())
    # return model
    JuMP.optimize!(model)
    try
        x = @. JuMP.value(x⁺) - JuMP.value(x⁻)
        return sparse(x)
    catch e
        println(e)
        x = fill(NaN, size(A, 2))
        return sparse(x)
    end
end
const bpd = basis_pursuit_denoising

function bpd_reweighting(A::AbstractMatrix, b::AbstractVector, δ::Real,
                                    reweighting!; maxiter = 8, min_decrease = 1e-4)
    x = bpd(A, b, δ)
    w = ones(length(x))
    for i in 2:maxiter
        reweighting!(w, x) # updates w
        xs = bpd(A, b, δ, w)
        if norm(xs-x) < min_decrease # stop if we don't make sufficient progress
            return xs
        end
        x .= xs
    end
    x
end

# δ is l2 noise threshold, ε is reweighting constant
# ε below 1e-3 can lead to convergence problems
function bpd_candes(A::AbstractMatrix, b::AbstractVector, δ::Real, ε::Real = δ; maxiter = 8)
    bpd_reweighting(A, b, δ, candes_function(ε), maxiter = maxiter)
end
function bpd_ard(A::AbstractMatrix, b::AbstractVector, δ::Real, ε::Real = δ^2; maxiter = 8)
    bpd_reweighting(A, b, δ, ard_function(A, ε), maxiter = maxiter)
end

# optimizer setup
function bpd_scs_model()
    JuMP.Model(optimizer_with_attributes(SCS.Optimizer,
                "verbose" => true,
                "eps" => 1e-6,
                "alpha" => 1.,
                "cg_rate" => 2.,
                "max_iters" => 5000))
end
function bpd_ecos_model()
    JuMP.Model(optimizer_with_attributes(ECOS.Optimizer, "verbose" => false))
end

################################ (F)ISTA ########################################
struct FISTA end
# TODO: use homotopy to select appropriate λ

# shrinkage and thresholding operator
shrinkage(x::Real, α::Real) = sign(x) * max(abs(x)-α, zero(x))

# TODO fast iterative shrinkage and hard-thresholding algorithm
# for weighted l1-norm minimization
function fista(A::AbstractMatrix, b::AbstractVector, λ::Real,
    x::AbstractVector = spzeros(size(A, 2)); maxiter::Int = 1024, stepsize::Real = 1e-2)
    w = fill(λ, size(x))
    fista(A, b, w, x, maxiter = maxiter, stepsize = stepsize)
end

function l1(x::AbstractVector, w::AbstractVector)
    length(x) == length(w) || throw(DimensionMismatch("length(x) ≠ length(w)"))
    n = zero(eltype(x))
    @simd for i in eachindex(x)
        @inbounds n += w[i] * abs(x[i])
    end
    return n
end

# TODO: stepsize selection
function ista(A, b, λ::Real, x = spzeros(size(A, 2)); maxiter::Int = 1024, stepsize::Real = 1e-2)
    ista(A, b, fill(λ, size(x)), x, maxiter = maxiter, stepsize = stepsize)
end

function ista(A::AbstractMatrix, b::AbstractVector, w::AbstractVector,
                                        x::AbstractVector = spzeros(size(A, 2));
                                        maxiter::Int = 1024, stepsize::Real = 1e-2)
    x = sparse(x)
    r(x) = b-A*x # residual
    f(x) = sum(abs2, r(x)) + l1(x, w)
    g(x) = A'r(x) # negative gradient
    α = stepsize
    fx = f(x)
    for i in 1:maxiter
        ∇ = g(x)
        @. x = shrinkage(x + 2α*∇, w*α)
        dropzeros!(x) # optimize sparse representation
    end
    return x
end


function fista(A::AbstractMatrix, b::AbstractVector, w::AbstractVector,
                                        x::AbstractVector = spzeros(size(A, 2));
                                        maxiter::Int = 1024, stepsize::Real = 1e-2)
    x = sparse(x)
    r(x) = b-A*x # residual
    f(x) = sum(abs2, r(x)) + l1(x, w)
    g(x) = A'r(x) # negative gradient
    α = stepsize
    tk = 1
    for i in 1:maxiter
        ∇ = g(x)
        tkn = (1 + sqrt(1 + 4tk^2)) / 2
        y = xkn + (tk - 1) / tkn * (xkn - xk)
        tk = tkn
        @. x = shrinkage(x + 2*∇, w*α)
        dropzeros!(x) # optimize sparse representation
    end
    return x
end
