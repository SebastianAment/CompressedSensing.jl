######################### Benchmarks
using BenchmarkTools
using CompressedSensing: rmp, rmps, sbl, synthetic_data

δ = 1e-2
A, x, b = perturbed_gaussian_data(n, m, k, δ) # defined in PhaseTransitions
a1(A, b) = rmp(A, b, δ)

# function benchmark(n, m, k)
#     # A, x, b = coherent_data(n, m, k)
#     # A, x, b = gaussian_data(n, m, k)
#     δ = 1e-2
#     a1 = (A, b) -> omp(A, b, δ)
#     a2 = (A, b) -> rp(A, b, δ)
#     a3 = (A, b) -> nrp(A, b, δ, normalize = true)
#     a4 = (A, b) -> nrp(A, b, δ, normalize = false)
#     a5 = (A, b) -> bpd(A, b, δ)
#     a6 = (A, b) -> bpd_ard(A, b, δ)
#
#     #
#     # algs = [a1, a2, a3]
#     # for a in algs
#     #     @a(A, b)
#     # end
#     suite = BenchmarkGroup()
#     suite["omp"] = @benchmarkable $a1($A, $b)
#     suite["rp"] = @benchmarkable $a2($A, $b)
#     suite["nrp"] = @benchmarkable $a3($A, $b)
#     suite["nrp no rs"] = @benchmarkable $a4($A, $b)
#     suite["bpd"] = @benchmarkable $a5($A, $b)
#     suite["pbd_ard"] = @benchmarkable $a6($A, $b)
#     suite
# end
#
# function run_benchmark()
#     n, m, k = 128, 256, 16
#     a = 1 # 1, 2, 3, ...
#     n, m, k = a .* (n, m, k)
#     b = benchmark(n, m, k)
#     r = run(b)
# end

# end RMPExperiments
