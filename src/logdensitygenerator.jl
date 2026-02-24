struct ModelLogDensity{F, G}
    dim::Int
    ℓ::F
    constrain::G
end

log_prob(m::ModelLogDensity, q) = m.ℓ(q)

@generated function process_params(q::Vector{Float64}, transforms::T) where {T<:Tuple}
    N = length(T.parameters)
    
    q_exprs = [:(transform(transforms[$i], q[$i])) for i in 1:N]
    lj_exprs = [:(log_abs_det_jacobian(transforms[$i], q[$i])) for i in 1:N]
    
    return quote
        q_constrained = tuple($(q_exprs...))
        
        log_jac = +($(lj_exprs...))
        
        return q_constrained, log_jac
    end
end

function build_log_joint(data, lpdf, transforms::Tuple)
    return function(q_unconstrained::Vector{Float64}) 
        
        q_constrained, log_jac = process_params(q_unconstrained, transforms)
        
        target = 0.0
        for x in data
            target += lpdf(x, q_constrained...) 
        end
        
        return target + log_jac
    end
end

## Reverse mode — one forward+backward pass, O(1) in dim
function ∇logp_reverse!(g::Vector{Float64}, ℓ::ModelLogDensity, q::Vector{Float64})
    fill!(g, 0.0)

    result = try
        Enzyme.autodiff(
            Enzyme.ReverseWithPrimal,
            log_prob,
            Enzyme.Active,
            Enzyme.Const(ℓ),
            Enzyme.Duplicated(q, g)
        )
    catch e
        @error "Enzyme autodiff failed" exception = e
        nothing
    end

    if result === nothing
        fill!(g, 0.0)
        return -Inf, false
    end

    lp = result[2]

    if !isfinite(lp) || any(isnan, g) || any(!isfinite, g)
        fill!(g, 0.0)
        return -Inf, false
    end

    return lp, true
end

## Batched forward mode — one widened forward pass, all partials at once
function ∇logp_forward!(g::Vector{Float64}, ℓ::ModelLogDensity, q::Vector{Float64}, seeds)
    fill!(g, 0.0)

    result = try
        Enzyme.autodiff(
            Enzyme.ForwardWithPrimal,
            log_prob,
            Enzyme.Const(ℓ),
            Enzyme.BatchDuplicated(q, seeds)
        )
    catch e
        @error "Enzyme forward autodiff failed" exception = e
        nothing
    end

    if result === nothing
        fill!(g, 0.0)
        return -Inf, false
    end

    lp = result[2]
    derivs = result[1]
    @inbounds for i in eachindex(g)
        g[i] = derivs[i]
    end

    if !isfinite(lp) || any(isnan, g) || any(!isfinite, g)
        fill!(g, 0.0)
        return -Inf, false
    end

    return lp, true
end
