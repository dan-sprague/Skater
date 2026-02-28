UNDER CONSTRUCTION!!!!!

# PhaseSkate - Scalable, High Performance Bayesian Inference In Native Julia

A native Julia implementation of the Stan approach to Bayesian inference. Built for [Enzyme](https://github.com/EnzymeAD/Enzyme.jl). Skate along the logjoint with ease!

## Design Philosophy

1. Speed - I built PhaseSkate because I needed a tool that could handle the large, messy and complex data+models of Biology. To do this, PhaseSkate was built **for** Enzyme with minimal external dependencies, with an emphasis on type stability and pure LPDF functions.
2. Clarity - the `@spec` that defines the Skate DSL is a complete description of the information needed by a reader (and the compiler!!) to understand the model. What this means in practice is that the information flowing into the model is not scattered across an analysis script. The `@spec` block, while technical, is visually distinct, cohesive, and defines everything. Stan got this right.


Bayesian inference is an area where the Julia language should shine. HMC is intensive, CPU bound numerical computations of complex gradients and at its heart a physics simulation of a particle flying around phase-space. I built this package because I became increasingly upset that I always found myself returning to Stan.

DSL

```julia

include("src/bijections.jl")
include("src/utilities.jl")
include("src/logdensitygenerator.jl")
include("src/lpdfs.jl")
import Base.@kwdef

include("src/lang.jl")

@spec MixtureModel begin
    @constants begin ## constants and data
        N::Int
        K::Int
        x::Vector{Float64}
    end

    @params begin ### these parameters will be sampled
        theta = param(Vector{Float64}, K; simplex = true)
        mus = param(Vector{Float64}, K, ordered = true)
        sigma = param(Float64; lower = 0.0)
    end

    @logjoint begin

        
        target += dirichlet_lpdf(theta, 1.0) # mixture weights
        target += multi_normal_diag_lpdf(mus, 0.0, 10.0) # means
        target += normal_lpdf(sigma, 0.0, 5.0) # shared var 

        # Likelihood
        for i in 1:N
            target += log_mix(theta) do j
                normal_lpdf(x[i], mus[j], sigma)
            end
        end
    end
end


K = 2
N = 50
x_data = vcat(randn(25) .- 2.0, randn(25) .+ 2.0)

d = MixtureModel_DataSet(N=N, K=K, x=x_data)
m = make_mixturemodel(d);

samples = sample(m, 1000; ϵ = 0.01, L = 10)

```
