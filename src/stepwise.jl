######################## Relevance Matching Pursuit ###########################
# equivalent to consecutive runs of forward and backward regression
# δ is the largest marginal increase/decrease in residual norm before
# rmp with outer iterations
function rmp(A::AbstractMatOrFac, b::AbstractVector, δ::Real,
                                    maxiter::Int = 1, x = spzeros(size(A, 2)))
    n = size(A, 1)
    xt = copy(x) # to keep track of changes in x
    P = StepwiseRegression(A, b, x.nzind)
    for i in 1:maxiter
        # foward stage
        for _ in 1:n
            forward_step!(P, x, 0, δ) || break # breaks if no change occured
        end
        !(xt ≈ x) || break # break outer loop if x is not changing anymore
        xt .= x
        # backward stage
        for _ in nnz(x):-1:1
            backward_step!(P, x, Inf, δ) || break # breaks if no change occured
        end
        !(xt ≈ x) || break # break outer loop if x is not changing anymore
        xt .= x
    end
    return x
end

# max_ε is the residual norm tolerance
# δ is the largest marginal increase/decrease in residual norm before
# either forward or backward algorithm terminates
# k is the desired sparsity of the solution
function rmp(A::AbstractMatOrFac, b::AbstractVector,
            k::Int, x = spzeros(size(A, 2)))
    n = size(A, 1)
    P = StepwiseRegression(A, b, x.nzind)
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
function foba(A::AbstractMatOrFac, b::AbstractVector, δ::Real, x = spzeros(size(A, 2));
                                                            isfast::Val = Val(true))
    n = size(A, 1)
    P = StepwiseRegression(A, b, x.nzind)
    for i in 1:n
        forward_step!(P, x, 0, δ) || break
        max_δ = sqrt(maximum(P.δ²)) # this is the change in residual norm of last forward step
        while backward_step!(P, x, Inf, max_δ/2, isfast) end # FoBa only takes backward steps if they increase the norm by half of decrease of forward algorithm
    end
    return x
end
