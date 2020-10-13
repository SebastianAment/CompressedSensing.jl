include("AISTATS2021_Experiments.jl")

δ = 1e-2
droptol = 1e-3
algorithms = [(A, b) -> omp(A, b, δ),
            (A, b) -> fr(A, b, 0, δ),
            (A, b) -> rmp(A, b, δ),
            (A, b) -> rmp(A, b, δ, 4), # rmp with outer iterations
            (A, b) -> foba(A, b, δ),
            (A, b) -> rmps(A, b, δ),
            (A, b) -> fsbl(A, b, δ),
            (A, y) -> droptol!(bpd(A, y, δ), droptol),
            (A, y) -> droptol!(bpd_ard(A, y, δ), droptol)
            ]

algnames = ["OMP", "FR", "RMP", "RMP+", "FoBa", "RMP_σ", "FSBL", "BP", "BP ARD"]

############### recovery sweep for fixed undersampling ratio ###################
m = 128
nexp = 1024
subsampling_fractions = [.5]
n = m÷2
nalg = length(algorithms)

################################################################################
sparsity_fractions = collect(1:n÷4) ./ n # sparsity coefficients
data_generator(n, m, k) = perturbed_coherent_data(n, m, k, δ/2)
success = zeros(nalg, nexp, 1, length(sparsity_fractions))
P = PhaseTransitionExperiment(m, nexp, subsampling_fractions, sparsity_fractions,
            algorithms, data_generator, success)
println("running perturbed coherent")
run!(P)

if doh5
    filename = "AISTATS2021_RecoverySweeps_NoisyCoherent"
    saveh5(P, algnames, δ, droptol, filename)
end

################################################################################
sparsity_fractions = collect(1:n÷4) ./ n # sparsity coefficients
data_generator(n, m, k) = coherent_data(n, m, k)
success = zeros(nalg, nexp, 1, length(sparsity_fractions))
P = PhaseTransitionExperiment(m, nexp, subsampling_fractions, sparsity_fractions,
            algorithms, data_generator, success)
println("running noiseless coherent")
run!(P)

if doh5
    filename = "AISTATS2021_RecoverySweeps_NoiselessCoherent"
    saveh5(P, algnames, δ, droptol, filename)
end

################################################################################
sparsity_fractions = collect(1:n÷2) ./ n # sparsity coefficients
data_generator(n, m, k) = perturbed_gaussian_data(n, m, k, δ/2)
success = zeros(nalg, nexp, 1, length(sparsity_fractions))
P = PhaseTransitionExperiment(m, nexp, subsampling_fractions, sparsity_fractions,
            algorithms, data_generator, success)
println("running perturbed gaussian")
run!(P)

doh5 = true
if doh5
    filename = "AISTATS2021_RecoverySweeps_NoisyGaussian"
    saveh5(P, algnames, δ, droptol, filename)
end

################################################################################
sparsity_fractions = collect(1:n÷2) ./ n # sparsity coefficients
data_generator(n, m, k) = gaussian_data(n, m, k)
success = zeros(nalg, nexp, 1, length(sparsity_fractions))
P = PhaseTransitionExperiment(m, nexp, subsampling_fractions, sparsity_fractions,
            algorithms, data_generator, success)
println("running noiseless gaussian")
run!(P)

if doh5
    filename = "AISTATS2021_RecoverySweeps_NoiselessGaussian"
    saveh5(P, algnames, δ, droptol, filename)
end
