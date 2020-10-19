################### stepwise regression with replacement (SRR) #################
# one-lookahead greedy two-step algorithm, based on stepwise regression algorithms
function srr(A::AbstractMatrix, b::AbstractVector, k::Int, δ::Real = 1e-12,
                                            x = spzeros(eltype(A), size(A, 2));
                                            maxiter = 4k, initialization::Int = 1,
                                            l::Int = 1)
    P = StepwiseRegression(A, b, x.nzind)
    # initialize support with k atoms using
    if initialization == 1 # oblivious algorithm
        oblivious_acquisition!(P, x, k-nnz(x)) # initialize with k largest inner products
    elseif initialization == 2 # forward regression
        for _ in 1:k
            update!(P, x)
        end
    elseif initialization == 3 # random indices
        random_acquisition!(P, x, k-nnz(x)) # initialize with k largest inner products
    end
    resnorm = norm(residual!(P, x))
    for i in 1:maxiter
        oldnorm = resnorm
        for _ in 1:l
            forward_step!(P, x, 0, 0) || break # add unless we found a solution
        end
        while nnz(x) > k # delete until we have an active set of size k
            backward_step!(P, x, Inf, Inf)
        end
        resnorm = norm(residual!(P, x))
        if resnorm ≤ δ || oldnorm ≤ resnorm # until convergence or stationarity
            break
        end
    end
    return x
end

@inline function argmaxinner!(P::StepwiseRegression, k::Int)
    Ar = P.A' * P.r
    @. Ar = abs(Ar)
    partialsortperm(Ar, 1:k, rev = true)
end

################################## Subspace Pursuit ############################
struct SubspacePursuit{T, AT<:AbstractMatrix{T}, B<:AbstractVector{T}} <: AbstractMatchingPursuit{T}
    A::AT
    b::B
    k::Int # maximum number of non-zeros

    # temporary storage
    r::B # residual
    Ar::B # inner products between measurement matrix and residual
    Ai::AT # space for A[:, x.nzind] and its qr factorization
end
const SP = SubspacePursuit

function SP(A::AbstractMatrix, b::AbstractVector, k::Integer)
    2k > length(b) && error("2k = $(2k) > $(length(b)) = length(b) is invalid for Subspace Pursuit")
    n, m = size(A)
    T = eltype(A)
    r, Ar = copy(b), zeros(T, m)
    Ai = zeros(T, (n, 2k))
    SP(A, b, k, r, Ar, Ai)
end

# returns indices of k atoms with largest inner products with residual
# could use threshold on P.Ar for adaptive stopping
# this is oblivious algorithm applied to residual
sp_index!(P::SP, k::Int = P.k) = argmaxinner!(P, k)
function sp_acquisition!(P::SP, x, k::Int = P.k)
    residual!(P, x)
    i = sp_index!(P, k)
    @. x[i] = NaN
    solve!(P, x)
end
# TODO: could pre-allocate nz arrays to be of length 2K
# TODO: could add efficient qr updating
function update!(P::SP, x::AbstractVector = spzeros(size(P.A, 2)), ε::Real = 0.)
    nnz(x) == P.k || throw("nnz(x) = $(nnz(x)) ≠ $(P.k) = k")
    sp_acquisition!(P, x)
    i = partialsortperm(abs.(x.nzval), 1:nnz(x)-P.k) # find the smallest atoms
    sort!(i) # for deleteat! to work, indices need to be sorted
    deleteat!(x.nzind, i)
    deleteat!(x.nzval, i)
    solve!(P, x) # optimize all active atoms
end

# calculates k-sparse approximation to Ax = b via subspace pursuit
# could also stop if indices are same between iterations
function sp(A::AbstractMatrix, b::AbstractVector, k::Int, δ::Real = 1e-12; maxiter = 16k)
    P = SP(A, b, k)
    x = spzeros(size(A, 2))
    sp_acquisition!(P, x, P.k)
    resnorm = norm(residual!(P, x))
    for i in 1:maxiter
        oldnorm = resnorm
        update!(P, x)
        resnorm = norm(residual!(P, x))
        if resnorm ≤ δ || oldnorm ≤ resnorm
            break
        end
    end
    return x
end

# ordinary least squares solve
@inline function solve!(P::SP, x::AbstractVector, b::AbstractVector = P.b)
    F = factorize!(P, x)
    ldiv!(x.nzval, F, b)     # optimize all active atoms
end

############################# OMP with replacement #############################
struct OMPR{T, AT<:AbstractMatrix{T}, B<:AbstractVector{T}, FT} <: AbstractMatchingPursuit{T}
    A::AT
    b::B
    k::Int
    # l # number of atoms to replace

    # temporary storage
    r::B # residual
    Ar::B # inner products between measurement matrix and residual
    QA::AT # (k, m)
    AiQR::FT # updatable QR factorization of Ai
end

# similar to RMP, or OMP?
function OMPR(A::AbstractMatrix, b::AbstractVector, k::Int)
    n, m = size(A)
    T = eltype(A)
    r, Ar = zeros(T, n), zeros(T, m)
    # AiQR = UpdatableQR(reshape(A[:, 1], :, 1))
    AiQR = PUQR(reshape(A[:, 1], :, 1))
    QA = zeros(T, (n, m))
    remove_column!(AiQR) # start with empty factorization
    OMPR(A, b, k, r, Ar, QA, AiQR)
end

# η is stepsize
function update!(P::OMPR, x::AbstractVector, η::Real = 1.)
    nnz(x) == P.k || throw("nnz(x) = $(nnz(x)) ≠ $(P.k) = k")
    residual!(P, x)
    copy!(P.Ar, x)
    mul!(P.Ar, P.A', P.r, η, 1)
    begin # argmax of abs(P.Ar) over i not in x.nzind     # argmax( [abs(ar) for ar in P.Ar]
        m = 0.
        i = 0
        for (j, Arj) in enumerate(P.Ar)
            if j ∉ x.nzind
                fArj = abs(Arj)
                if fArj > m
                    m = fArj
                    i = j
                end
            end
        end
    end

    if i == 0
        return x
    end

    # add non-zero index to active set
    x[i] = NaN
    qr_i = findfirst(==(i), x.nzind) # index in qr where new atom should be added

    # least-squares solve for active atoms (this would be two-stage algorithm)
    # ldiv!(x.nzval, P.AiQR, P.b)

    # gradient descent for active atoms
    @. x.nzval = P.Ar[x.nzind]

    j = argmin(abs, x.nzval) # delete index
    deleteat!(x.nzind, j) # delete the value from array
    deleteat!(x.nzval, j)

    if qr_i ≠ j # update qr factorization using Givens rotations
        a = @view P.A[:, i]
        add_column!(P.AiQR, a, qr_i)
        remove_column!(P.AiQR, j)
    end

    # least-squares solve for active atoms
    ldiv!(x.nzval, P.AiQR, P.b)
    return x
end

# k is desired sparsity level
# l is cardinality of maximum replacement per iteration, l = k corresponds to sp
function ompr(A::AbstractMatrix, b::AbstractVector, k::Int, δ::Real,
                                x = spzeros(size(A, 2)); maxiter = size(A, 1))
    P = OMPR(A, b, k)
    if nnz(x) < P.k # make sure support set is of size k
        @. x = 0
        dropzeros!(x)
        oblivious_acquisition!(P, x, P.k)
    end
    resnorm = norm(residual!(P, x))
    for i in 1:maxiter
        oldnorm = resnorm
        update!(P, x)
        resnorm = norm(residual!(P, x))
        if resnorm ≤ δ || oldnorm ≤ resnorm
            break
        end
    end
    return x
end
