## LEAPFROG INTEGRATOR
include("transformed_densities.jl")


struct PhaseSpacePoint
    q::Vector{Float64}
    p::Vector{Float64}
    grad_q::Vector{Float64}
end

### Q REPRESENTS THE POSITION IN PHASE SPACE OF THE PARAMETERS, P REPRESENTS THE MOMENTUM OF THE PARAMETERS. IN HMC, WE SAMPLE MOMENTUM FROM A GAUSSIAN DISTRIBUTION, AND THEN SIMULATE THE DYNAMICS OF THE SYSTEM TO PROPOSE NEW SAMPLES.

function leapfrog!(state::PhaseSpacePoint,grad,ϵ = 0.1)


end