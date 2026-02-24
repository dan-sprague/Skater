UNDER CONSTRUCTION!!!!!

# Skate - Scalable, High Performance Bayesian Inference In Native Julia

A native Julia implementation of the Stan approach to Bayesian inference. Built for [Enzyme](https://github.com/EnzymeAD/Enzyme.jl). Skate along the logjoint with ease!

## Design Philosophy

1. Clarity - the `@spec` that defines the Skate DSL is a complete description of the information needed by a reader (and the compiler!!) to completely understand the model. What this means in practice is that the information flowing into the model is not scattered across an analysis script. The `@spec` block, while technical, is visually distinct, cohesive, and defines everything. Stan got this right.
2. Type stability, up front - Skate requires the user to be explicit about types and dimensions up front. This package is built on the belief that this is good practice and saves time and confusion for the user. No need to wonder if your code is type stable or not!
3. Fast, native Julia - Skate was built to match or exceed Stan's sampling speed while remaining 100% Julia.

Bayesian inference is practically the raison d'être for the Julia language. Intensive, CPU bound numerical computations of complex gradients and at its heart a physics simulation of a particle flying around phase-space. I built this package because I became increasingly upset that I always found myself returning to Stan.

DSL

```julia
include("src/bijections.jl")
include("src/utilities.jl")
include("src/logdensitygenerator.jl")
include("src/lpdfs.jl")
import Base.@kwdef

include("src/lang.jl")

@spec MixtureModel begin
    @data begin ## constants and data
        N::Int
        K::Int
        x::Vector{Float64}
    end

    @params begin ### these parameters will be sampled
        theta = param(Vector{Float64}, K; simplex = true)
        mus = param(Vector{Float64}, K)
        sigma = param(Float64; lower = 0.0)
    end

    @logjoint begin
\

        # Priors
        # Dirichlet prior on mixing weights
        target += dirichlet_lpdf(theta, 1.0)

        # Mean priors 
        for j in 1:K
            target += normal_lpdf(mus[j], 0.0, 10.0)
        end

        # Prior on sigma
        target += normal_lpdf(sigma, 0.0, 5.0)

        # Likelihood
        for i in 1:N
            target += log_mix(theta) do j
                normal_lpdf(x[i], mus[j], sigma)
            end
        end
    end
end

K = 3
N = 50
x_data = vcat(randn(25) .- 2.0, randn(25) .+ 2.0)

d = MixtureModel_DataSet(N=N, K=K, x=x_data)
m = make_mixturemodel(d)
println("dim = ", m.dim)
# K-1 (simplex) + K (mus) + 1 (sigma) = 2 + 3 + 1 = 6
println("expected dim = ", (K-1) + K + 1)

q = randn(m.dim)
lp = log_prob(m, q)
println("log_prob = ", lp)
println("isfinite = ", isfinite(lp))

using Test
import Enzyme
@allocations m.ℓ(q)

g = zeros(length(q))
∇logp!(g, m, q)
g
```
