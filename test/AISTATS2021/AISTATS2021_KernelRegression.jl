using Statistics
using StatsBase
using LinearAlgebra
using SparseArrays
using CompressedSensing
using CompressedSensing: omp, fr, foba, fsbl, bpd, bpd_ard, sbl, rmp, rmps, colnorms
using DelimitedFiles
using CSV
using Kernel

function subsample_data(A, y, ntrain = 512)
    n, m = size(A)
    itrain = sample(1:n, ntrain, replace = false)
    At = @view A[itrain, :] # training data
    yt = @view y[itrain]
    At, yt
end

housing = true
superconductor = false
################# Import Data
if housing
    file = "UCI Data/BostonHousing/housing.data"
    A = readdlm(file)
    y = A[:, end] # median value
    A = A[:, 1:end-1] # attributes
elseif superconductor
    file = "UCI Data/Superconductor/train.csv"
    f = CSV.File(file)

    n = length(f)
    m = length(f[1]) - 1
    A = zeros(n, m) # chemical features
    y = zeros(n) # critical temperature
    for i in 1:n
        for j in 1:m
            A[i,j] = f[i][j]
        end
        # A[i, m] = 1 # constant offset doesn't change performance
        y[i] = f[i][end]
    end
    A, y = subsample_data(A, y, n ÷ 2)
end

n, m = size(A)
A .-= mean(A, dims = 1) # center features
A ./= colnorms(A)' # normalize features
y .-= mean(y)
y ./= norm(y)
println(size(A))

################ set up algorithms for comparison
algorithms = [
            (A, b, δ) -> fr(A, b, 0, δ),
            rmp,
            (A, b, δ) -> rmp(A, b, δ, 4), # rmp with outer iterations
            # foba,
            rmps,
            fsbl,
            (A, y, δ) -> droptol!(sparse(sbl(A, y, δ)), δ/10),
            ]

algnames = ["FR", "RMP", "RMP+", "RMP_σ", "FSBL", "SBL"]
nalgs = length(algorithms)

function run(A, y, algorithms, δ, algnames = algnames)

    weights = []
    for (i, f) in enumerate(algorithms)
        println(algnames[i])
        @time push!(weights, f(A, y, δ))
    end
    residuals = zeros(size(weights))
    sparsity = zeros(Int, size(weights))
    for (i, w) in enumerate(weights)
        residuals[i] = norm(A*w-y)
        sparsity[i] = nnz(w)
    end
    return weights, residuals, sparsity
end

############### linear regression
δ = .05 # .05, .2, .3
weights, residuals, sparsity = run(A, y, algorithms, δ)
display(residuals)
display(sparsity)

############### kernel regression
println("kernel regression")
l = .05 # length scale
k = Kernel.Lengthscale(Kernel.MaternP(1), l)
K = Kernel.gramian(k, A')
println(size(K))
K = Matrix(K)
δ = .05 # .05, .2, .3
weights, residuals, sparsity = run(K, y, algorithms, δ)
display(residuals)
display(sparsity)
