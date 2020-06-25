using LinearAlgebra
using CompressedSensing
using StatsBase: sample, mean
using SparseArrays
using Base.Threads

# pursuit algorithms
using CompressedSensing: mp, omp, rp, nrp, sp, greedy_sbl
# lp based solvers
using CompressedSensing: basispursuit, candes_reweighting, ard_reweighting

# IDEA: use TaylorSeries + overloading to define
# non-trivial matrix functions, like sin etc.
# could instead use ApproxFun and Chebyshev approximation
function gaussian_data(n = 32, m = 64, k = 16; ε = 1e-6, σ = 0., normalized = true)
    A = randn(n, m)
    if normalized
        A .-= (1-ε) * mean(A, dims = 1) # can't subtract mean completely for preconditioner to be invertible
        A ./= sqrt.(sum(abs2, A, dims = 1))
    end
    x = k_sparse_vector(m, k) # make sure they are above ε
    b = A*x
    # if σ > 0
    #     @. b += σ^2 * randn()
    # end
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
    x = k_sparse_vector(m, k)
    b = A*x
    A, x, b
end

function k_sparse_vector(m, k)
    x = spzeros(m)
    ind = sort!(sample(1:m, k, replace = false))
    @. x[ind] = randn()
    x
end

function samesupport(x, y)
    sort!(x.nzind) == sort!(y.nzind)
end

function experiment(algorithms, data_generator)
    A, x, b = data_generator()
    xalg = Vector{typeof(x)}(undef, length(algorithms))
    for (i, f) in enumerate(algorithms)
        xalg[i] = f(A, b)
    end
    x, xalg
end

function get_algorithms(k::Int)
    ε = 1e-4 # tolerance for support recovery
    a1 = (A, b) -> droptol!(omp(A, b, k), ε) # runs omp and drops any coefficient above ε
    a2 = (A, b) -> droptol!(nrp(A, b, ε, normalize = true), ε)
    a3 = (A, b) -> droptol!(basispursuit(A, b), ε)
    w_ε = 1e-2 #
    a4 = (A, b) -> droptol!(ard_reweighting(A, b, w_ε^2, maxiter = 10), ε)
    a5 = (A, b) -> droptol!(candes_reweighting(A, b, w_ε, maxiter = 10), ε)
    a6 = (A, b) -> sp(A, b, k, ε)
    δ = 1e-6
    a7 = (A, b) -> droptol!(rp(A, b, δ), ε)
    a8 = (A, b) -> droptol!(nrp(A, b, ε, normalize = false), ε)
    algorithms = [a1, a2, a3, a4, a5, a6, a7, a8]
    names = ["omp", "nrp", "bp", "bp_ard", "bp_candes", "sp", "rp", "nrp no rescale"]
    algorithms, names
end

using Optimization: bpd, bpd_ard, bpd_candes
function noisy_algorithms(k::Int, δ::Real = 1e-2)
    ε = 1e-4 # tolerance for support recovery
    a1 = (A, b) -> droptol!(omp(A, b, δ), ε) # runs omp and drops any coefficient above ε
    a2 = (A, b) -> droptol!(nrp(A, b, δ), ε)
    a3 = (A, b) -> droptol!(bpd(A, b, δ), ε)
    w_ε = 1e-2
    a4 = (A, b) -> droptol!(bpd_ard(A, b, δ, w_ε^2, maxiter = 10), ε)
    # a5 = (A, b) -> droptol!(bpd_candes(A, b, δ, w_ε, maxiter = 10), ε)
    a6 = (A, b) -> sp(A, b, k, δ)
    a7 = (A, b) -> droptol!(rp(A, b, δ), ε)
    a8 = (A, b) -> droptol!(nrp(A, b, ε, normalize = false), ε)

    algorithms = [a1, a2, a3, a4, a6, a7, a8] #a5, a6, a7]
    names = ["omp", "nrp", "bp", "bp_ard", "sp", "rp", "nrp no rescale"] #"bp_candes", "sp",  "rp"]
    algorithms, names
end

function mp_algorithms(k::Int, δ::Real = 1e-2)
    ε = 1e-4
    a1 = (A, b) -> droptol!(omp(A, b, δ), ε) # runs omp and drops any coefficient above ε
    a2 = (A, b) -> droptol!(nrp(A, b, δ, normalize = true), ε)
    a7 = (A, b) -> droptol!(rp(A, b, δ), ε)
    algorithms = [a1, a2, a7]
    names = ["omp", "nrp", "rp"]
    algorithms, names
end

# @threads
using Base.Threads
function driver(n, m, k, nexp)
    begin
        algorithms, algnames = get_algorithms(k)
        data_generator(x...) = gaussian_data(n, m, k)
        # data_generator(x...) = coherent_data(n, m, k)
    end
    # begin
    #     # algorithms, algnames = noisy_algorithms(k)
    #     algorithms, algnames = mp_algorithms(k)
    #     δ = 1e-2
    #     data_generator(x...) = perturbed_gaussian_data(n, m, k, δ)
    #     # data_generator(x...) = perturbed_coherent_data(n, m, k, δ)
    # end

    nalg = length(algorithms) # number of algorithms to benchmark
    sparsity = zeros(nalg, nexp)
    correct_support = zeros(nalg, nexp)
    residual = zeros(nalg, nexp)
    @threads for i in 1:nexp # threads
        x, xalg = experiment(algorithms, data_generator)
        sparsity[:,i] = nnz.(xalg) # records number of non-zero coefficient of solution
        f(y) = samesupport(x, y)
        correct_support[:, i] .= f.(xalg)
        resnorm(y) = norm(x-y)
        residual[:, i] .= resnorm.(xalg)
    end
    correct_support, residual, sparsity
end

# n = 128
# m = 256
# k = 64
# @time nsuccess, residual = driver(n, m, k, σ, nexp)
using HDF5
function run_experiment(save = false)
    n = 64
    m = 128
    algorithms, algnames = get_algorithms(1)
    # algorithms, algnames = noisy_algorithms(1)
    # algorithms, algnames = mp_algorithms(1)
    if save
        # filename = "perturbed_gaussian_sparse_recovery.h5"
        # filename = "perturbed_coherent_sparse_recovery.h5"
        # filename = "perturbed_coherent_mp_rescaling.h5"
        # filename = "perturbed_gaussian_mp_rescaling.h5"
        # filename = "coherent_sparse_recovery.h5"
        filename = "gaussian_sparse_recovery.h5"
        savefile = h5open(filename, "w")
        write(savefile, "algorithm order", algnames)
    end

    # karr = 1:1:7 # for coherent noisy
    # karr = [1, 2, 4, 8, 12, 16, 20, 24, 28] # for coherent noiseless
    # karr = 15:15
    karr = 8:4:32
    nalg = length(algorithms)
    nsuccess = zeros(nalg, length(karr))
    for (i, k) in enumerate(karr)
        println(i)
        nexp = 1024
        correct_support, residual, sparsity = driver(n, m, k, nexp)
        # nsuccess[:, i] .= reshape(mean(nsupport, dims = 2), :)
        if save
            g_create(savefile, "$k")
            write(savefile["$k"], "correct_support", correct_support)
            write(savefile["$k"], "residual", residual)
            write(savefile["$k"], "sparsity", sparsity)
        end
    end
    close(savefile)
    nsuccess
end

using Statistics
# println(mean(nsupport, dims = 2))
# println(mean(residual, dims = 2))
# println(median(residual, dims = 2))

using Plots
plotlyjs()
# pyplot()
function plot_recovery(file)
    datanames = filter(!=("algorithm order"), names(file))
    nk = length(datanames)
    psuccess = zeros(3, nk)
    stdsuccess = zeros(3, nk)
    karr = zeros(Int, nk)
    nexp = zeros(Int, nk)
    for (i, name) in enumerate(datanames)
        nsupport = read(file[name], "correct_support")
        psuccess[:, i] = mean(nsupport, dims = 2)
        stdsuccess[:, i] = std(nsupport, dims = 2)
        karr[i] = tryparse(Int, name)
        nexp[i] = size(nsupport, 2)
    end
    println(karr)
    ind = sortperm(karr)
    karr = karr[ind]
    psuccess = psuccess[:, ind]
    nexp = nexp[1] # assuming nexp is constant
    plot(karr, permutedims(psuccess), label = permutedims(["omp", "rp", "bp"]),
        xlabel = "k", ylabel = "p(success)", linewidth = 2.,
        ribbon = permutedims(stdsuccess/sqrt(nexp)),
        xtickfont = font(18),
        ytickfont = font(18),
        xguidefont = font(18),
        yguidefont = font(18),
        legendfont = font(18))
end

function plot_time()
    karr = range(12, stop = 32, step = 2)
    t = range(0, stop = 1, length = length(karr))
    tomp = [0.301, .410, .537, .698, .878, 1.146, 1.379, 1.649, 1.931, 2.142, 2.556] # results in ms for 128 by 256 system
    tbp = [35.039, 35.543, 35.674, 36.041, 35.621, 37.013, 37.546, 37.088, 39.646, 39.724, 38.566]
    tnrp = [0.378, .504, .629, .792, 1.026, 1.258, 1.548, 1.852, 2.152, 2.484, 2.969]
    plot(karr, [tomp, tnrp, tbp], label = permutedims(["omp", "rp", "bp"]),
        xlabel = "k",
        ylabel = "ms",
        yscale = :log10,
        linewidth = 2.,
        xtickfont = font(18),
        ytickfont = font(10),
        xguidefont = font(18),
        yguidefont = font(18),
        legendfont = font(18))
end

# Plots.scalefontsizes(2)

function report_recovery(file)
    datanames = filter(!=("algorithm order"), names(file))
    nk = length(datanames)
    algnames = read(file["algorithm order"])
    nalg = length(algnames)
    psuccess = zeros(nalg, nk)
    stdsuccess = zeros(nalg, nk)
    karr = zeros(Int, nk)
    # algnames = []
    median_error = zeros(nalg, nk)
    for (i, name) in enumerate(datanames)
        # push!(algnames, name)
        correct_support = read(file[name], "correct_support")
        error_i = read(file[name], "residual")
        median_error[:, i] = median(error_i, dims=2)
        psuccess[:, i] = mean(correct_support, dims = 2)
        stdsuccess[:, i] = std(correct_support, dims = 2)
        karr[i] = tryparse(Int, name)
    end
    # println(karr)
    # println(median_error[3,:]) # bp
    # println(median_error[7,:]) # rp
    println(size(median_error))
    ind = sortperm(karr)
    karr = karr[ind]
    psuccess = psuccess[:, ind]
    # algnames = ["omp", "nrp", "bp", "ard", "candes", "ssp", "rp"]
    # algnames = ["omp", "nrp", "bp", "ard", "ssp", "rp"]

    plot(karr, permutedims(psuccess), label = permutedims(algnames),
        xlabel = "k", ylabel = "p(success)", linewidth = 2.,
        ribbon = 2permutedims(stdsuccess)/sqrt(1024))
    gui()
    # plot(karr, permutedims(psuccess), label = permutedims(algnames),
    #     xlabel = "k", ylabel = "p(success)", linewidth = 2.)
end

########################## Benchmarks
using BenchmarkTools
function bench(n, m, k)
    # A, x, b = coherent_data(n, m, k)
    # A, x, b = gaussian_data(n, m, k)
    δ = 1e-2
    A, x, b = perturbed_gaussian_data(n, m, k, δ)
    a1 = (A, b) -> omp(A, b, δ)
    a2 = (A, b) -> rp(A, b, δ)
    a3 = (A, b) -> nrp(A, b, δ, normalize = true)
    a4 = (A, b) -> nrp(A, b, δ, normalize = false)
    a5 = (A, b) -> bpd(A, b, δ)
    a6 = (A, b) -> bpd_ard(A, b, δ)

    #
    # algs = [a1, a2, a3]
    # for a in algs
    #     @a(A, b)
    # end
    suite = BenchmarkGroup()
    suite["omp"] = @benchmarkable $a1($A, $b)
    suite["rp"] = @benchmarkable $a2($A, $b)
    suite["nrp"] = @benchmarkable $a3($A, $b)
    suite["nrp no rs"] = @benchmarkable $a4($A, $b)
    suite["bpd"] = @benchmarkable $a5($A, $b)
    suite["pbd_ard"] = @benchmarkable $a6($A, $b)
    suite
end

function run_bench()
    n = 128
    m = 256
    k = 16
    # n = 256
    # m = 512
    # k = 32
    # n = 512
    # m = 1024
    # k = 64
    b = bench(n, m, k)
    r = run(b)
end
