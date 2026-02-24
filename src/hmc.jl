## LEAPFROG INTEGRATOR

### Q REPRESENTS THE POSITION IN PHASE SPACE OF THE PARAMETERS, P REPRESENTS THE MOMENTUM OF THE PARAMETERS. IN HMC, SAMPLE MOMENTUM FROM A GAUSSIAN DISTRIBUTION, AND THEN SIMULATE THE DYNAMICS OF THE SYSTEM TO PROPOSE NEW SAMPLES.

function leapfrog!(q, p, g, model, ∇logp!, ϵ = 0.1)

    ∇logp!(g, model, q)
    p .+= (ϵ / 2) .* g

    q .+= ϵ .* p

    ∇logp!(g, model, q)
    p .+= (ϵ / 2) .* g

    return nothing

end

struct PhaseSpacePoint
    q::Vector{Float64}
    p::Vector{Float64}
end
struct HMCState
    curr::PhaseSpacePoint
    proposal::PhaseSpacePoint
    grad::Vector{Float64}
end

function hamiltonian(q, p, model)
    return -log_prob(model, q) + 0.5 * sum(abs2, p)
end

## Nesterov dual averaging for step size adaptation
mutable struct DualAveraging
    μ::Float64       # log(10 * ε₀)
    log_ε̄::Float64   # smoothed log step size
    H̄::Float64       # smoothed acceptance statistic
    m::Int           # iteration count
    δ::Float64       # target acceptance rate
    γ::Float64
    t₀::Float64
    κ::Float64
end

function DualAveraging(ε₀::Float64; δ = 0.8)
    DualAveraging(log(10.0 * ε₀), 0.0, 0.0, 0, δ, 0.05, 10.0, 0.75)
end

function adapt!(da::DualAveraging, α)
    da.m += 1
    m = da.m
    w = 1.0 / (m + da.t₀)
    da.H̄ = (1.0 - w) * da.H̄ + w * (da.δ - α)
    log_ε = da.μ - sqrt(m) / da.γ * da.H̄
    mk = m^(-da.κ)
    da.log_ε̄ = mk * log_ε + (1.0 - mk) * da.log_ε̄
    return exp(log_ε)
end

adapted_ε(da::DualAveraging) = exp(da.log_ε̄)

function _hmc_step!(HMC, model, ϵ, L)
    randn!(HMC.curr.p)
    HMC.proposal.q .= HMC.curr.q
    HMC.proposal.p .= HMC.curr.p
    H_current = hamiltonian(HMC.curr.q, HMC.curr.p, model)
    for _ in 1:L
        leapfrog!(HMC.proposal.q, HMC.proposal.p, HMC.grad, model, ∇logp!, ϵ)
    end
    H_proposal = hamiltonian(HMC.proposal.q, HMC.proposal.p, model)
    α = min(1.0, exp(H_current - H_proposal))
    if rand() < α
        HMC.curr.q .= HMC.proposal.q
        HMC.curr.p .= HMC.proposal.p
    end
    return α
end

## note -- ∇logp! handles zeroing of gradient buffer
function sample(model, num_samples; ϵ = 0.1, L = 10, warmup = 1000)
    printstyled("⚙  Compiling gradient...\n"; color=:yellow, bold=true)
    _, ok = ∇logp!(zeros(model.dim), model, randn(model.dim))
    ok || error("Gradient compilation failed")
    printstyled("✓  Gradient ready\n"; color=:green, bold=true)

    HMC = HMCState(
        PhaseSpacePoint(zeros(Float64, model.dim), zeros(Float64, model.dim)),
        PhaseSpacePoint(zeros(Float64, model.dim), zeros(Float64, model.dim)),
        zeros(Float64, model.dim)
    )

    ## ── Warmup: adapt step size via dual averaging ──
    printstyled("~  Warmup "; color=:yellow, bold=true)
    printstyled("$warmup"; color=:white, bold=true)
    printstyled(" iterations  ϵ₀=$ϵ\n"; color=:yellow)

    da = DualAveraging(Float64(ϵ))
    ϵ_curr = Float64(ϵ)
    for i in 1:warmup
        α = _hmc_step!(HMC, model, ϵ_curr, L)
        ϵ_curr = adapt!(da, α)
    end
    ϵ = adapted_ε(da)

    printstyled("✓  Adapted ϵ = "; color=:green, bold=true)
    printstyled("$(round(ϵ; sigdigits=4))\n"; color=:white, bold=true)

    ## ── Sampling ──
    printstyled("~  Sampling "; color=:cyan, bold=true)
    printstyled("$num_samples"; color=:white, bold=true)
    printstyled(" samples  ϵ=$(round(ϵ; sigdigits=4))  L=$L\n"; color=:cyan)

    T = Base.promote_op(model.constrain, Vector{Float64})
    samples = Vector{T}(undef, num_samples)
    progress_interval = max(1, num_samples ÷ 10)
    @inbounds for i in 1:num_samples
        _hmc_step!(HMC, model, ϵ, L)
        samples[i] = model.constrain(HMC.curr.q)
        if i % progress_interval == 0
            pct = 100i ÷ num_samples
            printstyled("   $(lpad(pct, 3))%  "; color=:light_black)
            printstyled("$i/$num_samples\n"; color=:light_black)
        end
    end
    printstyled("✓  Done\n"; color=:green, bold=true)
    return samples
end
