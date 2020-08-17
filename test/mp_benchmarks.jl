bench = true
using BenchmarkTools
if bench
    m = 1024
    n = m ÷ 2
    k = n ÷ 8
    A, x, b = sparse_data(n = n, m = m, k = k)

    δ = 1e-6
    @test nnz(omp(A, b, δ)) == k
    @test nnz(rmp(A, b, δ)) == k

    # @btime mp($A, $b, $k)
    @btime omp($A, $b, $δ)
    # @btime omp($A, $b, $k)
    # # @btime sp($A, $b, $k)
    @btime rmp($A, $b, $δ, rescale = true)

    # Notes:
    # RMP consumes roughly 2x memory to OMP

    # @time xomp = omp(A, b, k)
    # @time xrp = rmp(A, b)
    # @time xbp = basispursuit(A, b)

end
