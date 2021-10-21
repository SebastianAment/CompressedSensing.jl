# exhaustive search for k-subset selection
function exhaustive(A::AbstractMatOrFac, b::AbstractVector, k::Int)
    n, m = size(A)
    x = zeros(k)
    indices = CartesianIndices(tuple((1:n for _ in 1:k)...))
    support = indices[1]
    min_r = Inf
    for i in indices
        Ai = @view A[:, i]
        x = Ai \ b
        r = b - Ai*x
        normr = norm(r)
        if normr < min_r
            min_r = normr
            support = i
        end
    end
    return Tuple(support) # tuple of indices
end
