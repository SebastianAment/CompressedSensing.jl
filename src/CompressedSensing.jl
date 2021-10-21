module CompressedSensing

using LinearAlgebra
using SparseArrays
using LinearAlgebraExtensions

# TODO: change this to new updatable QR implementation
# ADD "GeneralizedOMP"
# using LinearAlgebraExtensions: UpdatableQR, UQR, PUQR, add_column!, remove_column!
using UpdatableQRFactorizations
const PUQR = UQR

# using LinearAlgebraExtensions: Projection, AbstractMatOrUni, AbstractMatOrFac
const AbstractMatOrFac{T} = Union{AbstractMatrix{T}, Factorization{T}}
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
include("util.jl") # sparse data generator
include("oblivious.jl")
include("matchingpursuit.jl")
include("forward.jl")
include("backward.jl")
include("twostage.jl")
include("stepwise.jl")
include("sbl.jl")
include("basispursuit.jl")

end
