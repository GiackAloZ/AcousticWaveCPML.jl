using BenchmarkTools, HypothesisTests

"""
 Compute d, K and a parameters for CPML
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

    # distance from first non-CPML node
    dist = LinRange(1-shift, Kab_size-shift, Kab_size)
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

"""
 Gaussian source time function
"""
function gaussource1D( t::Vector{<:Real}, t0::Real, f0::Real )
    # boh = f0 .* (t-t0)
    # source = -8.0*boh.*exp( -boh.^2/(4.0*f0)^2 )
    boh= pi.*f0.*(t.-t0)
    source = -boh.*exp.( -boh.^2 )    
    return source
end


"""
 Ricker source time function
"""
function rickersource1D(t::Vector{<:Real}, t0::Real, f0::Real)    
    b = (pi*f0*(t.-t0)).^2
    w = (1.0.-2.0.*b).*exp.(.-b)
    return w
end

function check_trial(trial::BenchmarkTools.Trial, confidence=0.95, range=0.05)
    data = trial.times
    m = median(trial).time
    test = SignTest(data, m)
    ci_left, ci_right = confint(test, level=confidence)
    tol_left, tol_right = (m - m*range, m + m*range)
    return ci_left >= tol_left && ci_right <= tol_right, (ci_left / 1e9, ci_right / 1e9), (tol_left / 1e9, tol_right / 1e9)
end
