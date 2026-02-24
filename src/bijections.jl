abstract type Constraint end

struct IdentityConstraint <: Constraint end
transform(::IdentityConstraint, x) = x
log_abs_det_jacobian(::IdentityConstraint, x) = 0.0
grad_correction(::IdentityConstraint, x) = 1.0

# ── ExpBijection family: shared Jacobian and grad_correction ──────────────────
abstract type ExpBijection <: Constraint end
log_abs_det_jacobian(::ExpBijection, x) = x
grad_correction(::ExpBijection, x) = exp(x)

struct LowerBounded{T} <: ExpBijection lb::T end
LowerBounded(lb::Real) = LowerBounded{Float64}(Float64(lb))
transform(c::LowerBounded, x) = c.lb + exp(x)

struct UpperBounded{T} <: ExpBijection ub::T end
UpperBounded(ub::Real) = UpperBounded{Float64}(Float64(ub))
transform(c::UpperBounded, x) = c.ub - exp(x)

# ── Bounded (logistic sigmoid) ────────────────────────────────────────────────
struct Bounded{T} <: Constraint
    lb::T
    ub::T
    function Bounded(lb::T, ub::T) where T
        lb isa Real && ub isa Real && lb >= ub &&
            error("Lower bound must be less than upper bound")
        new{T}(lb, ub)
    end
end
Bounded(lb, ub) = Bounded(promote(lb, ub)...)
Bounded(lb::Real, ub::Real) = Bounded{Float64}(Float64(lb), Float64(ub))

_logistic(x) = x >= 0 ? 1.0 / (1.0 + exp(-x)) : exp(x) / (1.0 + exp(x))

function transform(c::Bounded, x)
    s = _logistic(x)
    c.lb + (c.ub - c.lb) * s
end

function log_abs_det_jacobian(c::Bounded, x)
    s = _logistic(x)
    log(c.ub - c.lb) + log(s) + log(1.0 - s)
end

function grad_correction(c::Bounded, x)
    s = _logistic(x)
    (c.ub - c.lb) * s * (1.0 - s)
end
