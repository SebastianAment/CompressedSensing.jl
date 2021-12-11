module CompressedSensing

using LinearAlgebra
using SparseArrays

using LazyInverses
using WoodburyFactorizations
using UpdatableQRFactorizations

const AbstractMatOrFac{T} = Union{AbstractMatrix{T}, Factorization{T}}

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
