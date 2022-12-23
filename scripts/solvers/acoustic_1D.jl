using Plots, Plots.Measures
using BenchmarkTools
using Printf

default(size=(1000, 600), framestyle=:box,grid=false,margin=20pt)

include("../utils.jl")

# folders for results
DOCS_FLD = joinpath(dirname(dirname(@__DIR__)), "simulations")

# Disable interactive visualization
ENV["GKSwstype"]="nul"

"""
    update_ψ!(ψ_l, ψ_r, pcur,
              halo, _dx,
              a_x_hl, a_x_hr,
              b_K_x_hl, b_K_x_hr)

Update the CPML ψ arrays (left and right) using the coefficients provided by parameters and current pressure `pcur`.
"""
@views function update_ψ!(ψ_l, ψ_r, pcur,
                          halo, _dx,
                          a_x_hl, a_x_hr,
                          b_K_x_hl, b_K_x_hr)
    nx, _... = size(pcur)
    for i = 1:halo+1
        ii = i + nx - halo - 2  # shift for right boundary pressure indices
        # left boundary
        ψ_l[i] = b_K_x_hl[i] * ψ_l[i] + a_x_hl[i]*(pcur[ i+1] - pcur[ i])*_dx
        # right boundary
        ψ_r[i] = b_K_x_hr[i] * ψ_r[i] + a_x_hr[i]*(pcur[ii+1] - pcur[ii])*_dx
    end
end

"""
    update_p!(pold, pcur, pnew, halo, fact, _dx, _dx2,
              ψ_l = nothing, ψ_r = nothing,
              ξ_l = nothing, ξ_r = nothing,
              a_x_l = nothing, a_x_r = nothing,
              b_K_x_l = nothing, b_K_x_r = nothing)

Update the pressure array `pnew` by using old pressure values, the `fact` array with prescaled velocity and CPML ψ and ξ arrays.
"""
@views function update_p!(pold, pcur, pnew, halo, fact, _dx, _dx2,
                          ψ_l = nothing, ψ_r = nothing,
                          ξ_l = nothing, ξ_r = nothing,
                          a_x_l = nothing, a_x_r = nothing,
                          b_K_x_l = nothing, b_K_x_r = nothing)
    nx, _... = size(pcur)
    for i = 2:nx-1
        d2p_dx2 = (pcur[i+1] - 2.0*pcur[i] + pcur[i-1])*_dx2

        if i <= halo+1
            # left boundary
            dψ_dx = (ψ_l[i] - ψ_l[i-1])*_dx
            ξ_l[i-1] = b_K_x_l[i-1] * ξ_l[i-1] + a_x_l[i-1] * (d2p_dx2 + dψ_dx)
            damp = fact[i] * (dψ_dx + ξ_l[i-1])
        elseif i >= nx - halo
            # right boundary
            ii = i - (nx - halo) + 2
            dψ_dx = (ψ_r[ii] - ψ_r[ii-1])*_dx
            ξ_r[ii-1] = b_K_x_r[ii-1] * ξ_r[ii-1] + a_x_r[ii-1] * (d2p_dx2 + dψ_dx)
            damp = fact[i] * (dψ_dx + ξ_r[ii-1])
        else
            damp = 0.0
        end

        # update pressure
        pnew[i] = 2.0 * pcur[i] - pold[i] + fact[i] * (d2p_dx2) + damp
    end
end

"""
    inject_sources!(pnew, dt2srctf, possrcs, it)

Inject sources onto the `pnew` array with waveforms in `dt2srctf` and positions `possrcs` for iteration `it`.
"""
@views function inject_sources!(pnew, dt2srctf, possrcs, it)
    _, nsrcs = size(dt2srctf)
    for i = 1:nsrcs
        isrc = possrcs[i,1]
        pnew[isrc] += dt2srctf[it,i]
    end
end

"""
    kernel!(
        pold, pcur, pnew, fact, _dx, _dx2,
        halo, ψ_l, ψ_r, ξ_l, ξ_r,
        a_x_hl, a_x_hr, b_K_x_hl, b_K_x_hr,
        a_x_l, a_x_r, b_K_x_l, b_K_x_r,
        possrcs, dt2srctf, it
    )

Perform the timestep number `it` of the acoustic 1D computation on pressure and CPML arrays.

Return the pressure arrays swapped according to the following scheme:
- pold --> pnew
- pcur --> pold
- pnew --> pcur
"""
@views function kernel!(
    pold, pcur, pnew, fact, _dx, _dx2,
    halo, ψ_l, ψ_r, ξ_l, ξ_r,
    a_x_hl, a_x_hr, b_K_x_hl, b_K_x_hr,
    a_x_l, a_x_r, b_K_x_l, b_K_x_r,
    possrcs, dt2srctf, it
)
    update_ψ!(ψ_l, ψ_r, pcur,
              halo, _dx,
              a_x_hl, a_x_hr,
              b_K_x_hl, b_K_x_hr)
    update_p!(pold, pcur, pnew, halo, fact, _dx, _dx2,
              ψ_l, ψ_r,
              ξ_l, ξ_r,
              a_x_l, a_x_r,
              b_K_x_l, b_K_x_r)
    inject_sources!(pnew, dt2srctf, possrcs, it)

    return pcur, pnew, pold
end

"""
    acoustic1D(
        lx::Real,
        nt::Integer,
        vel::Vector{<:Real};
        halo::Integer = 20,
        rcoef::Real = 0.0001,
        do_vis::Bool = true,
        do_bench::Bool = false,
        nvis::Integer = 2,
        gif_name::String = "acoustic1D",
        plims::Vector{<:Real} = [-1e-1, 1e-1]
    )

Compute `nt` timesteps of the acoustic 1D wave equation with CPML boundary conditions on a model with size `lx` meters,
velocity field `vel`, number of CPML layers in each boundary `halo` and CPML reflection coeffiecient `rcoef`.

Return the last timestep pressure.

# Arguments
- `do_vis`: to plot visualization or not.
- `do_bench`: to perform a benchmark instead of the computation.
- `nvis`: frequency of timestep for visualization
- `gif_name`: name of the gif to save
- `plims`: pressure limits in visualizion plot
"""
@views function acoustic1D(
    lx::Real,
    nt::Integer,
    vel::Vector{<:Real};
    halo::Integer = 20,
    rcoef::Real = 0.0001,
    do_vis::Bool = true,
    do_bench::Bool = false,
    nvis::Integer = 2,
    gif_name::String = "acoustic1D",
    plims::Vector{<:Real} = [-1e-1, 1e-1]
)
    # Physics
    f0 = 8.0                            # dominating frequency [Hz]
    t0 = 1.2 / f0                       # activation time [s]
    # Derived physics
    vel_max = maximum(vel)              # maximum velocity [m/s]
    # Numerics
    nx            = length(vel)         # number of grid points
    npower        = 2.0
    K_max         = 1.0
    # Derived numerics
    dx = lx / (nx-1)                    # grid step size [m]
    dt = dx / vel_max                   # timestep size (CFL + Courant condition) [s]
    times = collect(range(0.0,step=dt,length=nt))   # time vector [s]
    # CPML numerics
    alpha_max        = 2.0*π*(f0/2.0)
    thickness_cpml_x = halo * dx
    d0_x             = -(npower + 1) * vel_max * log(rcoef) / (2.0 * thickness_cpml_x)
    a_x_l , a_x_r , b_K_x_l , b_K_x_r  = calc_Kab_CPML(halo,dt,npower,d0_x,alpha_max,K_max,"ongrd")
    a_x_hl, a_x_hr, b_K_x_hl, b_K_x_hr = calc_Kab_CPML(halo,dt,npower,d0_x,alpha_max,K_max,"halfgrd")
    # precomputations
    _dx = 1.0 / dx
    _dx2 = 1.0 / (dx^2)
    fact = (dt^2) .* vel.^2

    # assertions for stability
    @assert dx <= vel_max/(10.0 * f0)   # at least 10pts per wavelength
    
    # Array initialization

    # pressure arrays
    pold = zeros(nx)
    pcur = zeros(nx)
    pnew = zeros(nx)
    # CPML arrays
    ψ_l, ψ_r = zeros(halo+1), zeros(halo+1)
    ξ_l, ξ_r = zeros(halo), zeros(halo)
    # sources
    possrcs = zeros(Int,1,1)
    possrcs[1,1] = div(nx, 2, RoundUp)
    # source time functions
    dt2srctf = zeros(nt,1)
    dt2srctf[:,1] .= (dt^2) .* 1000 * rickersource1D(times, t0, f0)

    # benchmarking instead of actual computation
    if do_bench
        t_it = @belapsed $kernel!(
            $pold, $pcur, $pnew, $fact, $_dx, $_dx2,
            $halo, $ψ_l, $ψ_r, $ξ_l, $ξ_r,
            $a_x_hl, $a_x_hr, $b_K_x_hl, $b_K_x_hr,
            $a_x_l, $a_x_r, $b_K_x_l, $b_K_x_r,
            $possrcs, $dt2srctf, 1
        )
        # effective memory access [GB]
        A_eff = (
            2*( 4*(halo+1) + (halo+1) ) +                   # update_ψ!
            2*( 4*(halo+1) + (halo+1) ) +                   # update_p! (CPML layers)
            (4*nx + nx)                                     # update_p! (all points)
        ) * sizeof(Float64) / 1e9
        # effective memory throughput [GB/s]
        T_eff = A_eff / t_it
        @printf("Time: %1.3e sec, Teff = %1.3f GB/s \n", t_it, T_eff)
        return nothing
    end

    # create results folders
    if do_vis
        mkpath(DOCS_FLD)
    end

    # time loop
    anim = Animation()
    for it=1:nt
        pold, pcur, pnew = kernel!(
            pold, pcur, pnew, fact, _dx, _dx2,
            halo, ψ_l, ψ_r, ξ_l, ξ_r,
            a_x_hl, a_x_hr, b_K_x_hl, b_K_x_hr,
            a_x_l, a_x_r, b_K_x_l, b_K_x_r,
            possrcs, dt2srctf, it
        )

        # visualization
        if do_vis && (it % nvis == 0)
            # update plims
            plims[1] = min(plims[1], minimum(pnew))
            plims[2] = max(plims[2], maximum(pnew))
            # sources positions
            scatter((possrcs[:,1] .- 1) .* dx, fill(0.0, size(possrcs, 1)); markershape=:star, color=:red, label="sources")
            # CPML boundaries
            plot!(fill(halo * dx, 2), plims; color=:grey, linestyle=:dot, label="CPML boundary")
            plot!(fill(lx - (halo * dx), 2), plims; color=:grey, linestyle=:dot, label=:none)
            # plot pressure
            plot!(0:dx:lx, pnew;
                  ylim=plims, xlims=(0,lx), xlabel="lx", ylabel="pressure", label="pressure", color=:blue,
                  title="1D Acoustic CPML\n(nx=$(nx), halo=$(halo), rcoef=$(rcoef))")
            frame(anim)
        end
    end
    # save visualization
    if do_vis
        gif(anim, joinpath(DOCS_FLD, "$(gif_name).gif"))
    end

    return pcur
end
