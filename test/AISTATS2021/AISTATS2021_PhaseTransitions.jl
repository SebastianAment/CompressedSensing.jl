include("AISTATS2021_Experiments.jl")

################################################################################
# droptol = 1e-3 # coefficients which can be dropped from solution of bp-based methods
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


δ = 1e-2
data_generator(n, m, k) = perturbed_gaussian_data(n, m, k, δ/2)
P = PhaseTransitionExperiment(m = 64, nexp = 1, nsample = 8, nsparse = 8,
                        algorithms = algorithms, data_generator = data_generator)
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
