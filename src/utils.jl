using BenchmarkTools, HypothesisTests

@doc raw"""
    calc_Kab_CPML(halo::Integer,
                  dt::Float64, npower::Float64, d0::Float64,
                  alpha_max_pml::Float64, K_max_pml::Float64,
                  onwhere::String)

Compute K, a and b parameters for CPML with `halo` layers.

# Arguments
- `dt`: times step size.
- `npower`: CPML power coefficient.
- `d0`: damping factor.
- `alpha_max_pml`: frequency dependent CPML coefficient.
- `K_max_pml`: maximum value for K coefficient.
- `onwhere`: "ongrd" for coefficients in nodal points, "halfgrd" for staggerd coefficients (in between nodal points).
"""
function calc_Kab_CPML(halo::Integer,
                       dt::Float64, npower::Float64, d0::Float64,
                       alpha_max_pml::Float64, K_max_pml::Float64,
                       onwhere::String)
    Kab_size = halo
    # shift for half grid coefficients
    if onwhere == "halfgrd"
        Kab_size += 1
        shift = 0.5
    elseif onwhere == "ongrd"
        shift = 0.0
    else
        error("Wrong onwhere parameter!")
    end

    # distance from edge node
    dist = collect(LinRange(0-shift, Kab_size-shift-1, Kab_size))
    if onwhere == "halfgrd"
        dist[1] = 0
    end
    normdist_left = reverse(dist) ./ halo
    normdist_right = dist ./ halo

    d_left = d0 .* (normdist_left .^ npower)
    alpha_left = alpha_max_pml .* (1.0 .- normdist_left)
    K_left = 1.0 .+ (K_max_pml - 1.0) .* (normdist_left .^ npower)
    b_left = exp.( .-(d_left ./ K_left .+ alpha_left) .* dt )
    a_left = d_left .* (b_left .- 1.0) ./ (K_left .* (d_left .+ K_left .* alpha_left))
    b_K_left = b_left ./ K_left

    d_right = d0 .* (normdist_right .^ npower)
    alpha_right = alpha_max_pml .* (1.0 .- normdist_right)
    K_right = 1.0 .+ (K_max_pml - 1.0) .* (normdist_right .^ npower)
    b_right = exp.( .-(d_right ./ K_right .+ alpha_right) .* dt )
    a_right = d_right .* (b_right .- 1.0) ./ (K_right .* (d_right .+ K_right .* alpha_right))
    b_K_right = b_right ./ K_right

    return a_left, a_right, b_K_left, b_K_right
end

@doc raw"""
gaussource1D(t::Real, t0::Real, f0::Real)    

First derivative of gaussian source function for current time `t``, activation time `t0` and dominating frequency `f0`.
"""
function gaussource1D(t::Real, t0::Real, f0::Real)
    return (t-t0) * exp(-((pi * f0 * (t - t0))^2))
end


@doc raw"""
    rickersource1D(t::Real, t0::Real, f0::Real)    

Ricker source time function for current time `t``, activation time `t0` and dominating frequency `f0`.
"""
function rickersource1D(t::Real, t0::Real, f0::Real)
    return (1 - 2 * (pi * f0 * (t - t0))^2) * exp(-((pi * f0 * (t - t0))^2))
end

"""
    check_trial(trial::BenchmarkTools.Trial, confidence=0.95, range=0.05)

Check that a `BenchmarkTools.Trial` satisfies the following property:
- the `confidence*100`% confidence interval lies inside the +-`range*100`% of the median. 

Return true if the property is satisfied, the confidence interval in seconds, the +- range % of median in seconds and the median in seconds.
"""
function check_trial(trial::BenchmarkTools.Trial, confidence=0.95, range=0.05)
    data = trial.times
    m = median(trial).time
    test = SignTest(data, m)
    ci_left, ci_right = confint(test, level=confidence)
    tol_left, tol_right = (m - m*range, m + m*range)
    return ci_left >= tol_left && ci_right <= tol_right, (ci_left / 1e9, ci_right / 1e9), (tol_left / 1e9, tol_right / 1e9), m / 1e9
end

@doc raw"""
Type representing a multi-source configuration for a wave propagation shot.
"""
struct Sources
    n::Integer
    positions::Matrix{<:Real}
    t0s::Vector{<:Real}
    srctfs::Vector{<:Function}
    freqdomain::Real

    @doc raw"""
        Sources(
            positions::Matrix{<:Real},
            t0s::Vector{<:Real},
            srctfs::Vector{<:Function},
            freqdomain::Real
        )

    Create a single shot wave propagation source configuration from source positions, time-functions and a frequency domain.
    """
    function Sources(
        positions::Matrix{<:Real},
        t0s::Vector{<:Real},
        srctfs::Vector{<:Function},
        freqdomain::Real
    )
        @assert size(positions, 1) > 0 "There must be at least one source!"
        @assert size(positions, 1) == length(t0s) == length(srctfs) "Number of sources do not match between positions and time-functions!"
        new(size(positions, 1), positions, t0s, srctfs, freqdomain)
    end
end

@doc raw"""
Type representing a multi-receiver configuration for a wave propagation shot.
"""
mutable struct Receivers
    n::Integer
    positions::Matrix{<:Real}
    seismograms::Matrix{<:Real}

    @doc raw"""
        Receivers(positions::Matrix{<:Real})

    Create a single shot wave propagation receivers configuration from receivers positions.
    """
    function Receivers(positions::Matrix{<:Real})
        @assert size(positions, 1) > 0 "There must be at least one receiver!"
        new(size(positions, 1), positions, zeros(Float64, (0,0)))
    end
end


