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
    @constants begin
        N::Int
        K::Int
        x1::Vector{Float64}
        x2::Vector{Float64}
    end

    @params begin
        theta = param(Vector{Float64}, K; simplex = true)
        mu1 = param(Vector{Float64}, K, ordered = true)
        mu2 = param(Vector{Float64}, K)
        sigma = param(Float64; lower = 0.0)
    end

    @logjoint begin
        target += dirichlet_lpdf(theta, 1.0)
        target += multi_normal_diag_lpdf(mu1, 0.0, 10.0)
        target += multi_normal_diag_lpdf(mu2, 0.0, 10.0)
        target += normal_lpdf(sigma, 0.0, 5.0)

        for i in 1:N
            target += log_mix(theta) do j
                normal_lpdf(x1[i], mu1[j], sigma) + normal_lpdf(x2[i], mu2[j], sigma)
            end
        end
    end
end


# ── Generate 2D mixture data ─────────────────────────────────────────────────
K = 2
N = 500
true_mu1 = [-2.0, 2.0]
true_mu2 = [1.0, -1.0]
true_sigma = 0.5

components = [rand() < 0.5 ? 1 : 2 for _ in 1:N]
x1_data = [randn() * true_sigma + true_mu1[components[i]] for i in 1:N]
x2_data = [randn() * true_sigma + true_mu2[components[i]] for i in 1:N]

d = MixtureModel_DataSet(N=N, K=K, x1=x1_data, x2=x2_data)
m = make_mixturemodel(d);

using Test
@time samples =  sample(m, 1000; ϵ = 0.1, L = 10, warmup = 100)


# ── Turing comparison ────────────────────────────────────────────────────────
using Turing
using Bijectors: ordered
using FillArrays
using LinearAlgebra
using Distributions: MixtureModel, MvNormal

x_turing = hcat([[x1_data[i], x2_data[i]] for i in 1:N]...)  # 2×N matrix

@model function gmm_marginalized(x, ::Val{K}) where {K}
    D, N = size(x)
    mu1 ~ ordered(MvNormal(Zeros(K), 10.0^2 * I))
    mu2 ~ MvNormal(Zeros(K), 10.0^2 * I)
    w ~ Dirichlet(K, 1.0)
    σ ~ truncated(Normal(0, 5), 0, Inf)
    dists = [MvNormal([mu1[k], mu2[k]], σ^2 * I) for k in 1:K]
    mix = MixtureModel(dists, w)
    for n in 1:N
        x[:, n] ~ mix
    end
end

model = gmm_marginalized(x_turing, Val(K))
@time chain = Turing.sample(model, HMC(0.1, 10), 1000)