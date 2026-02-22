abstract type Transformation end

struct IdentityTransformation <: Transformation end
struct LogTransformation <: Transformation end

transform(::IdentityTransformation, x) = x
transform(::LogTransformation, x) = exp(x)

log_abs_det_jacobian(::IdentityTransformation, x) = 0.0
log_abs_det_jacobian(::LogTransformation, x) = x

grad_correction(::IdentityTransformation, x) = 1.0
grad_correction(::LogTransformation, x) = exp(x)

abstract type Density end
struct PointDensity <: Density 
    lpdf::Function
end
struct ProductDensity <: Density 
    lpdf::Function
end


struct TransformedLogDensity{T, F, D}
    data::D
    lpdf::F
    transforms::T 
    buffer::Vector{Float64} 
end

function (model::TransformedLogDensity)(q_unconstrained)
    logp = 0.0
    
    for i in eachindex(q_unconstrained)
        model.buffer[i] = transform(model.transforms[i], q_unconstrained[i])
        logp += log_abs_det_jacobian(model.transforms[i], q_unconstrained[i])
    end

    for x in model.data
        logp += model.lpdf(model.buffer, x)
    end
    
    return logp
end

function _likelihood(lpdf::T,θ,data) where T <: PointDensity
    logp = 0.0
    for x in data
        logp += lpdf(θ, x)
    end
    return logp
end


function _likelihood(lpdf::T,θ,data) where T <: ProductDensity
    lpdf(θ, data)
end