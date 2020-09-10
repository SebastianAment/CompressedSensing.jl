######################### Oblivious Algorithm ##################################
# chooses the k features most correlated with the target
function oblivious(A::AbstractMatrix, b::AbstractVector, k::Int)
    inner = abs.(A'b)
    nzind = partialsortperm(inner, 1:k, rev = true)
    x = spzeros(size(b))
    x[nzind] = @view(A[:, nzind]) \ b
    return x
end
