module CompressedSensing

using LinearAlgebra
using SparseArrays
using LinearAlgebraExtensions
using LinearAlgebraExtensions: UpdatableQR, UQR, PUQR, add_column!, remove_column!
using LinearAlgebraExtensions: Projection, AbstractMatOrUni, AbstractMatOrFac
using LazyInverse: inverse
using WoodburyIdentity

using Base.Threads
using StatsBase: sample, mean

using JuMP
using JuMP: Model, optimizer_with_attributes, @variable, @objective, @constraint
using JuMP: SecondOrderCone
using Clp # for linear program
using ECOS # ECOS is interior point method (more accurate but slower on large problems)
# using SCS # operator splitting method, fast but not high accuracy

abstract type Update{T} end
(U::Update)(x) = update!(U, x)

# Compressed Sensing algorithms
include("matchingpursuit.jl")
include("basispursuit.jl")
include("sbl.jl")
include("util.jl") # sparse data generator

end
