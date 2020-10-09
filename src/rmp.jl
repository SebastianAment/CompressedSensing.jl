######################## Relevance Matching Pursuit ###########################
# TODO: benchmark! and test against other algs
# provide option which gives k-sparse result
# consecutive runs of forward and backward regression

# δ is the largest marginal increase/decrease in residual norm before
function rmp(A::AbstractMatrix, b::AbstractVector,
            δ::Real, x = spzeros(size(A, 2)))
    n = size(A, 1)
    P = StepwiseRegression(A, b)
    for _ in 1:n
        forward_step!(P, x, 0, δ) || break # breaks if no change occured
    end
    P = FastBackwardRegression(P)
    for _ in nnz(x):-1:1
        backward_step!(P, x, Inf, δ) || break # breaks if no change occured
    end
    return x
end

# max_ε is the residual norm tolerance
# δ is the largest marginal increase/decrease in residual norm before
# either forward or backward algorithm terminates
# k is the desired sparsity of the solution
function rmp(A::AbstractMatrix, b::AbstractVector,
            k::Int, x = spzeros(size(A, 2)))
    n = size(A, 1)
    P = StepwiseRegression(A, b)
    for _ in 1:n
        forward_step!(P, x, 0, 0) || break
    end
    for _ in nnz(x):-1:k+1
        backward_step!(P, x, Inf, Inf) || break
    end
    return x
end

# see Adaptive Forward-Backward Greedy Algorithm for Sparse Learning with Linear Models
# terminates if algorithm can't decrease residual norm by more than δ
function foba(A::AbstractMatrix, b::AbstractVector, δ::Real = 1e-6, x = spzeros(size(A, 2)))
    n = size(A, 1)
    P = StepwiseRegression(A, b)
    for _ in 1:n
        forward_step!(P, x, 0, δ) || break
        max_δ = maximum(P.δ) # this is the change in residual norm of last forward step
        while backward_step!(P, x, Inf, max_δ/2) end # FoBa only takes backward steps if they increase the norm by half of decrease of forward algorithm
    end
    return x
end

# one-lookahead greedy two-step algorithm, based on RMP
function lmp(A::AbstractMatrix, b::AbstractVector, k::Int, δ::Real = 1e-12,
                            x = spzeros(eltype(A), size(A, 2)); maxiter = 2k)
    P = StepwiseRegression(A, b)
    # oblivious_acquisition!(P, x, k-nnz(x)) # initialize with k largest inner products
    random_acquisition!(P, x, k-nnz(x)) # initialize with k largest inner products
    resnorm = norm(residual!(P, x))
    for i in 1:maxiter
        oldnorm = resnorm
        forward_step!(P, x, 0, 0) # always add
        backward_step!(P, x, Inf, Inf) # always delete
        resnorm = norm(residual!(P, x))
        if resnorm ≤ δ || oldnorm ≤ resnorm # until convergence or stationarity
            break
        end
    end
    return x
end
