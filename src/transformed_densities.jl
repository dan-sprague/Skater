using Base: @kwdef 

abstract type Transformation end

struct IdentityTransformation <: Transformation end
struct LogTransformation <: Transformation end

transform(::IdentityTransformation, x) = x
transform(::LogTransformation, x) = exp(x)

log_abs_det_jacobian(::IdentityTransformation, x) = 0.0
log_abs_det_jacobian(::LogTransformation, x) = x

grad_correction(::IdentityTransformation, x) = 1.0
grad_correction(::LogTransformation, x) = exp(x)



@kwdef struct TransformedLogDensity{T<:Tuple, F, D}
    data::D
    lpdf::F
    transforms::T 
    buffer::Vector{Float64} 
end

function (model::TransformedLogDensity)(q_unconstrained::AbstractVector{T}) where {T}
    q_constrained = map(transform, model.transforms, q_unconstrained)
    
    log_jac = zero(T)
    for i in eachindex(model.transforms)
        log_jac += log_abs_det_jacobian(model.transforms[i], q_unconstrained[i])
    end

    log_lik = zero(T)
    for x in model.data
        log_lik += model.lpdf(x, q_constrained...)
    end
    
    return log_jac + log_lik
end

function ∇logp!(model::TransformedLogDensity, q_unconstrained)
    fill!(model.buffer, 0.0)

    lp = try 
        model(q_unconstrained)
    catch
        return -Inf,false
    end

    if !isfinite(lp)
        fill!(model.buffer, 0.0)
        return -Inf,false
    end

    ok = try
        Enzyme.autodiff(
            Enzyme.set_runtime_activity(Enzyme.set_strong_zero(Enzyme.Reverse)),
            Enzyme.Const(model),   # Wrap the functor here to mark it as constant
            Enzyme.Active,         # The return value (the log-density) is what we differentiate
            Enzyme.Duplicated(q_unconstrained, model.buffer) # The ONLY argument to your model
        )
        true
    catch e
        @error "Enzyme autodiff failed" exception = e
        false
    end

    if !ok
        fill!(model.buffer, 0.0)
        return -Inf,false
    end

    if any(isnan,model.buffer) || any(!isfinite, model.buffer)
        fill!(model.buffer, 0.0)
        return -Inf,false
    end

    return lp, true
end