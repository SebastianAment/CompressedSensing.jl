# module RMPExperiments
# RelevanceMatchingPursuit Experiments
using LinearAlgebra
using CompressedSensing
using StatsBase: sample, mean
using SparseArrays
using Base.Threads
using HDF5

using Statistics
using Plots
# plotlyjs()
plotly()

# pursuit algorithms
using CompressedSensing: omp, fr, rmp, foba,
                        sbl, fsbl, rmps,
                        bp, bp_candes, bp_ard,
                        bpd, bpd_candes, bpd_ard

# IDEA: use TaylorSeries + overloading to define
# (approximations to) non-trivial matrix functions, like sin etc.
# could instead use ApproxFun and Chebyshev approximation
function gaussian_data(n::Int, m::Int, k::Int; ε = 1e-6, normalized = true)
    A = randn(n, m)
    if normalized
        A .-= (1-ε) * mean(A, dims = 1) # can't subtract mean completely for preconditioner to be invertible
        A ./= sqrt.(sum(abs2, A, dims = 1))
    end
    x = sparse_vector(m, k)
    b = A*x
    A, x, b
end

# perturbation of norm δ/2
function perturb!(b::AbstractVector, δ::Real)
    e = randn(size(b))
    e *= δ/(2norm(e))
    b .+= e # perturb
end

function perturbed_gaussian_data(n, m, k, δ::Real)
    A, x, b = gaussian_data(n, m, k)
    perturb!(b, δ)
    A, x, b
end

function perturbed_coherent_data(n, m, k, δ::Real)
    A, x, b = coherent_data(n, m, k)
    perturb!(b, δ)
    A, x, b
end

function coherent_data(n, m, k; normalized = true)
    U = randn(n, n)
    V = randn(n, m)
    S = Diagonal([1/i^2 for i in 1:n])
    A = U*S*V
    # normalize
    if normalized
        A ./= sqrt.(sum(abs2, A, dims = 1))
    end
    x = sparse_vector(m, k)
    b = A*x
    A, x, b
end

# creates a k-sparse vector with ±1 as entries
function sparse_vector(m::Int, k::Int, gaussian::Bool = false)
    x = spzeros(m)
    ind = sort!(sample(1:m, k, replace = false))
    x[ind] .= gaussian ? randn(k) : rand((-1, 1), k)
    return x
end

function samesupport(x::AbstractVector, y::AbstractVector)
    sort!(x.nzind) == sort!(y.nzind)
end

################################################################################
struct RecoveryExperiment{T<:Real, F, A, S<:AbstractArray{T}}
    n::Int # number of rows
    m::Int # number of columns
    k::Int # sparsity level
    nexp::Int # number of individual runs for each algorithm

    algorithms::A
    data_generator::F
    success::S
end

function RecoveryExperiment(noisy::Bool, coherent::Bool)
    n, m = 64, 128
    nexp = 128
    ε = 1e-4
    δ = noisy ? 1e-2 : 1e-6
    data_generator = coherent ? coherent_data : gaussian_data
    data_generator = ()->gaussian_data(n, m)
    if !coherent
        k = 8:4:32
    elseif noisy
        k = 1:1:7
    elseif !noisy
        k = [1, 2, 4, 8, 12, 16, 20, 24, 28]
    end
    return RecoveryExperiment(n, m, k, nexp, data_generator)
end

function run!(E::RecoveryExperiment)
    nalg = length(E.algorithms)
    # success = zeros(Bool, nalg, E.nexp)
    for j in 1:E.nexp
        A, x0, y = E.data_generator(E.n, E.m, E.k)
        for i in 1:nalg
            x = algorithms[i](A, y)
            E.success[i, j] = samesupport(x0, x)
        end
    end
    return E.success
end

struct PhaseTransitionExperiment{T, V<:AbstractVector{T}, A<:AbstractVector, D, S}
    m::Int # number of features
    nexp::Int # number of independent experiments
    subsampling_fractions::V # subsampling coefficients
    sparsity_fractions::V # sparsity coefficients
    algorithms::A
    data_generator::D
    success::S # nalg x nexp x nsample x nsparse
end

function PhaseTransitionExperiment(; m::Int = 128, nexp::Int = 128,
                            nsample::Int = 64, nsparse::Int = 64,
                            algorithms = [omp], data_generator = gaussian_data)
    subsampling_fractions = range(.1, 1, length = nsample)
    sparsity_fractions = range(0., 1., length = nsparse)
    nalg = length(algorithms)
    success = zeros(Bool, nalg, nexp, nsample, nsparse)
    PhaseTransitionExperiment(m, nexp, subsampling_fractions, sparsity_fractions,
                                algorithms, data_generator, success)
end

function saveh5(P::PhaseTransitionExperiment, algnames, δ, droptol)
    filename = "AISTATS2021_PhaseTransitionExperiment.h5"
    savefile = h5open(filename, "w")
    write(savefile, "m", P.m)
    write(savefile, "nexp", P.nexp)
    write(savefile, "subsampling_fractions", collect(P.subsampling_fractions))
    write(savefile, "sparsity_fractions", collect(P.sparsity_fractions))
    write(savefile, "algorithm names", algnames)
    write(savefile, "success", P.success)
    write(savefile, "data type", "perturbed gaussian")
    write(savefile, "noise threshold", δ)
    write(savefile, "droptol", droptol)
    close(savefile)
end

function readh5(P::PhaseTransitionExperiment, algnames)
    filename = "AISTATS2021_PhaseTransitionExperiment.h5"
    savefile = h5open(filename, "w")
    write(savefile, "m", P.m)
    write(savefile, "nexp", P.nexp)
    write(savefile, "subsampling_fractions", collect(P.subsampling_fractions))
    write(savefile, "sparsity_fractions", collect(P.sparsity_fractions))
    write(savefile, "algorithms", algnames)
    write(savefile, "success", P.success)
end

function run!(P::PhaseTransitionExperiment)
    nalg = length(P.algorithms)
    nsample, nsparse = length(P.subsampling_fractions), length(P.sparsity_fractions)
    for (i, δ) in enumerate(P.subsampling_fractions)
        n = round(Int, δ * P.m)
        @threads for j in eachindex(P.sparsity_fractions)
            println(i, j)
            ρ = P.sparsity_fractions[j]
            k = max(1, round(Int, ρ * n))
            E = RecoveryExperiment(n, P.m, k, P.nexp,
                                    P.algorithms, P.data_generator,
                                    @view P.success[:, :, i, j])
            run!(E)
        end
    end
    return P.success
end

function plot(P::PhaseTransitionExperiment)
    mean_success = mean(P.success, dims = 2) # average over experimental runs
    nalg = length(P.algorithms)
    for i in 1:nalg
        heatmap(P.subsampling_fractions, P.sparsity_fractions, mean_success[i, 1, :, :]',
        xlabel = "n/m", ylabel = "k/n", label = "name")
        gui()
    end
end

################################################################################
droptol = 1e-3 # coefficients which can be dropped from solution of bp-based methods
algorithms = [(A, y) -> omp(A, y, δ),
            (A, y) -> fr(A, y, 0, δ),
            (A, b) -> rmp(A, b, δ),
            # (A, b) -> foba(A, b, δ),
            (A, b) -> rmps(A, b, δ),
            (A, b) -> fsbl(A, b, δ)
            (A, y) -> droptol!(bpd(A, y, δ), droptol),
            (A, y) -> droptol!(bpd_ard(A, y, δ), droptol)
            ]

algnames = ["OMP", "FR", "RMP", "RMP_σ", "FSBL", "BP", "BP ARD"]

# success = zeros(length(algorithms), nexp)
# E = RecoveryExperiment(n, m, k, nexp, algorithms, gaussian_data, success)
# run!(E)
δ = 1e-2
data_generator(n, m, k) = perturbed_gaussian_data(n, m, k, δ/2)
P = PhaseTransitionExperiment(algorithms = algorithms)
success = run!(P)

doplot = false
if doplot
    plot(P)
end

doh5 = true
if doh5
    saveh5(P, algnames, δ, droptol)
end
