include("AISTATS2021_Experiments.jl")

############### recovery sweep for fixed undersampling ratio ###################
m = 128
nexp = 16
subsampling_fractions = [.5]
n = m÷2
sparsity_fractions = collect(1:2:n÷2) ./ n # sparsity coefficients
nalg = length(algorithms)

success = zeros(nalg, nexp, 1, length(sparsity_fractions))
P = PhaseTransitionExperiment(m, nexp, subsampling_fractions, sparsity_fractions,
            algorithms, data_generator, success)
run!(P)

doplot = false
if doplot
    plot(P)
end

doh5 = false
if doh5
    filename = "AISTATS2021_RecoverySweeps"
    saveh5(P, algnames, δ, droptol, filename)
end
