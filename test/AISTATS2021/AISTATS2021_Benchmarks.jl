######################### Benchmarks
include("AISTATS2021_Experiments.jl")
using BenchmarkTools

struct BenchmarkExperiment{V<:AbstractVector{Int}, A<:AbstractVector, N, T}
    m::V
    algorithms::A
    algnames::N
    time::T # record minimum observed execution time in nano seconds
end
function BenchmarkExperiment(m, algorithms, algnames)
    t_min = zeros(length(algorithms), length(m))
    return BenchmarkExperiment(m, algorithms, algnames, t_min)
end
function saveh5(P::BenchmarkExperiment)
    filename = "AISTATS2021_Benchmark.h5"
    savefile = h5open(filename, "w")
    write(savefile, "m", P.m)
    write(savefile, "algorithm names", algnames)
    write(savefile, "time (ns)", P.time)
    close(savefile)
end
function run!(E::BenchmarkExperiment)
    for j in eachindex(E.m)
        run!(E, j)
    end
    return E.time
end

function run!(E::BenchmarkExperiment, j::Int)
    m = E.m[j]
    n = m ÷ 2
    k = n ÷ 4
    δ = 1e-2
    A, x, b = perturbed_gaussian_data(n, m, k, δ/2)
    for (i, f) in enumerate(E.algorithms)
        E.time[i, j] = minimum(@benchmark($f($A, $b))).time
    end
    return E.time
end

δ = 1e-2
algorithms = [
            (A, b) -> omp(A, b, δ),
            (A, b) -> fr(A, b, 0, δ),
            (A, b) -> rmp(A, b, δ),
            (A, b) -> rmp(A, b, δ, 4) # rmp with outer iterations
            (A, b) -> foba(A, b, δ),
            (A, b) -> rmps(A, b, δ),
            (A, b) -> fsbl(A, b, δ),
            (A, y) -> droptol!(bpd(A, y, δ), droptol),
            (A, y) -> droptol!(bpd_ard(A, y, δ), droptol)
            ]
algnames = ["OMP", "FR", "RMP", "RMP+", "FoBa", "RMP_σ", "FSBL", "BP", "BP ARD"]

m = @. 256 * 2^(0:3)
E = BenchmarkExperiment(m, algorithms, algnames)
run!(E)
# saveh5(E)
