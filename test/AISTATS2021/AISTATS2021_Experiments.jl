# contains functions and types that are used for individual experiments:
# PhaseTransitions, RecoverySweeps, and Benchmarks
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

using CompressedSensing: perturb!, perturb, gaussian_data, coherent_data, samesupport

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

function run!(E::RecoveryExperiment)
    nalg = length(E.algorithms)
    for j in 1:E.nexp
        A, x0, y = E.data_generator(E.n, E.m, E.k)
        for i in 1:nalg
            x = algorithms[i](A, y)
            E.success[i, j] = samesupport(x0, x)
        end
    end
    return E.success
end

################################################################################
struct PhaseTransitionExperiment{T, V<:AbstractVector{T}, A<:AbstractVector, D, S}
    m::Int # number of features
    nexp::Int # number of independent experiments
    subsampling_fractions::V # subsampling coefficients
    sparsity_fractions::V # sparsity coefficients
    algorithms::A
    data_generator::D
    success::S # nalg x nexp x nsample x nsparse
end

function PhaseTransitionExperiment(; m::Int = 128, nexp::Int = 4,
                            nsample::Int = 16, nsparse::Int = 16,
                            algorithms = [omp], data_generator = gaussian_data)
    subsampling_fractions = range(.1, 1, length = nsample)
    sparsity_fractions = range(0., 1., length = nsparse)
    nalg = length(algorithms)
    success = zeros(Bool, nalg, nexp, nsample, nsparse)
    PhaseTransitionExperiment(m, nexp, subsampling_fractions, sparsity_fractions,
                                algorithms, data_generator, success)
end

function saveh5(P::PhaseTransitionExperiment, algnames, δ, droptol, filename)
    filename = "$filename.h5"
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

function Plots.plot(P::PhaseTransitionExperiment)
    _plot(P.subsampling_fractions, P.sparsity_fractions, P.success)
end

function _plot(subsampling_fractions::AbstractVector, sparsity_fractions::AbstractVector,
                success::AbstractArray)
    mean_success = mean(success, dims = 2) # average over experimental runs
    nsample = length(subsampling_fractions)
    nalg = size(success, 1)
    for i in 1:nalg
        if nsample > 1
            heatmap(subsampling_fractions, sparsity_fractions, mean_success[i, 1, :, :]',
            xlabel = "n/m", ylabel = "k/n", label = "name")
        elseif nsample == 1
            plot(sparsity_fractions, vec(mean_success[i, 1, 1, :]),
            xlabel = "k/m", ylabel = "probability of success", label = "name")
        end
        gui()
    end
end


# function RecoveryExperiment(noisy::Bool, coherent::Bool)
#     n, m = 64, 128
#     nexp = 128
#     ε = 1e-4
#     δ = noisy ? 1e-2 : 1e-6
#     data_generator = coherent ? coherent_data : gaussian_data
#
#     # data_generator = ()->gaussian_data(n, m)
#     if !coherent
#         k = 8:4:32
#     elseif noisy
#         k = 1:1:7
#     elseif !noisy
#         k = [1, 2, 4, 8, 12, 16, 20, 24, 28]
#     end
#     return RecoveryExperiment(n, m, k, nexp, data_generator)
# end
