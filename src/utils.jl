"""
 Compute d, K and a parameters for CPML
"""
function calc_Kab_CPML(npts::Integer,
                       halo::Integer,
                       dt::Float64, npower::Float64, d0::Float64,
                       alpha_max_pml::Float64,K_max_pml::Float64,
                       onwhere::String)
    Kab_size = halo
    if onwhere=="halfgrd"
        Kab_size += 1
        shift = 0.5
    else
        shift = 0.0
    end
    K_left,K_right  = ones(Kab_size), ones(Kab_size)
    b_left,b_right  = ones(Kab_size), ones(Kab_size)
    a_left,a_right  = zeros(Kab_size), zeros(Kab_size)

    # Left border
    for i=1:Kab_size
        ii = i - shift
        if ii <= Float64(halo)
            normdist = (halo - ii) / halo
        else
            continue
        end
        d = d0 * normdist^npower
        alpha =  alpha_max_pml * (1.0 - normdist)
        K_left[i] = 1.0 + (K_max_pml - 1.0) * normdist^npower
        b_left[i] = exp( - (d / K_left[i] + alpha) * dt )
        a_left[i] = d * (b_left[i]-1.0)/(K_left[i]*(d+K_left[i]*alpha))

    end
    # Right border
    for i=1:Kab_size
        ii = i-1 + (npts - halo - 1) - shift
        if ii >= Float64(npts - halo - 1)
            normdist = (ii - (npts - halo - 1)) / halo
        else
            continue
        end
        d = d0 * normdist^npower
        alpha =  alpha_max_pml * (1.0 - normdist)
        K_right[i] = 1.0 + (K_max_pml - 1.0) * normdist^npower
        b_right[i] = exp( - (d / K_right[i] + alpha) * dt )
        a_right[i] = d * (b_right[i]-1.0)/(K_right[i]*(d+K_right[i]*alpha))
    end

    return K_left,K_right,a_left,a_right,b_left,b_right
end

"""
 Ricker source time function
"""
function rickersource1D(t::Vector{<:Real}, t0::Real, f0::Real)    
    b = (pi*f0*(t.-t0)).^2
    w = (1.0.-2.0.*b).*exp.(.-b)
    return w
end
