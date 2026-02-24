struct ModelLogDensity{F}
    dim::Int
    ℓ::F
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

function ∇logp!(g::Vector{Float64}, ℓ::ModelLogDensity, q_unconstrained::Vector{Float64})
    fill!(g, 0.0)

    lp = try 
        log_prob(ℓ, q_unconstrained)
    catch
        return -Inf, false
    end

    if !isfinite(lp)
        fill!(g, 0.0)
        return -Inf, false
    end

    ok = try
        Enzyme.autodiff(
            Enzyme.set_runtime_activity(Enzyme.set_strong_zero(Enzyme.Reverse)),
            log_prob,  
            Enzyme.Active,         
            Enzyme.Const(ℓ),
            Enzyme.Duplicated(q_unconstrained, g)
        )
        true
    catch e
        @error "Enzyme autodiff failed" exception = e
        false
    end

    if !ok || any(isnan, g) || any(!isfinite, g)
        fill!(g, 0.0)
        return -Inf, false
    end

    return lp, true
end
