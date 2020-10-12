include("AISTATS2021_Experiments.jl")

################################################################################
droptol = 1e-3 # coefficients which can be dropped from solution of bp-based methods
algorithms = [(A, b) -> omp(A, b, δ),
            (A, b) -> fr(A, b, 0, δ),
            (A, b) -> rmp(A, b, δ),
            # (A, b) -> foba(A, b, δ),
            (A, b) -> rmps(A, b, δ),
            # (A, b) -> fsbl(A, b, δ)
            # (A, y) -> droptol!(bpd(A, y, δ), droptol),
            # (A, y) -> droptol!(bpd_ard(A, y, δ), droptol)
            ]

# algnames = ["OMP", "FR", "RMP", "RMP_σ", "FSBL", "BP", "BP ARD"]
algnames = ["OMP", "FR", "RMP", "RMP_σ"]

m = 128
nexp = 256
nsample = 128
subsampling_fractions = range(.1, 1., length = nsample)
nsparse = 128
sparsity_fractions = range(0., .7, length = nsparse) # sparsity coefficients
nalg = length(algorithms)
δ = 1e-2
data_generator(n, m, k) = perturbed_gaussian_data(n, m, k, δ/2)
success = zeros(nalg, nexp, nsample, nsparse)
P = PhaseTransitionExperiment(m, nexp, subsampling_fractions, sparsity_fractions,
            algorithms, data_generator, success)
success = run!(P)

doplot = false
if doplot
    plot(P)
end

doh5 = true
if doh5
    filename = "AISTATS2021_PhaseTransitions"
    saveh5(P, algnames, δ, droptol, filename)
end
