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
function ard_weights!(w, A, x, ε, iter = 8)
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

function ard_reweighting(A::AbstractMatrix, b::AbstractVector,
                                            ε = 1e-2; maxiter = 8)
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
# ε below 1e-3 is a pain for convergence
function bpd_candes(A::AbstractMatrix, b::AbstractVector, δ::Real, ε = 1e-2; maxiter = 8)
    bpd_reweighting(A, b, δ, candes_function(ε), maxiter = maxiter)
end
function bpd_ard(A::AbstractMatrix, b::AbstractVector, δ::Real, ε = 1e-2; maxiter = 8)
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
