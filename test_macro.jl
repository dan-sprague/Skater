include("src/bijections.jl")
include("src/utilities.jl")
include("src/logdensitygenerator.jl")
include("src/lpdfs.jl")
include("src/hmc.jl")

import Enzyme
import Base.@kwdef
using Random
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

        
        target += dirichlet_lpdf(theta, 1.0) # symmetric Dirichlet prior on mixture weights
        target += multi_normal_diag_lpdf(mus, 0.0, 10.0) # independent normal priors on component means
        target += normal_lpdf(sigma, 0.0, 5.0) # half-normal prior on component stddevs

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


