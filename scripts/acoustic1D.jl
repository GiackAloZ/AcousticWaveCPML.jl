using Plots
using BenchmarkTools
using Printf

include("utils.jl")

DOCS_FLD = joinpath(dirname(@__DIR__), "docs")

@views function update_ψ!(ψ_l, ψ_r, pcur,
                          halo,
                          a_x_hl, a_x_hr,
                          b_K_x_hl, b_K_x_hr)
    nx, _... = size(pcur)
    for i = 1:halo+1
        ii = i + nx - halo - 2  # shift for right boundary pressure indices
        # left boundary
        ψ_l[i] = b_K_x_hl[i] * ψ_l[i] + a_x_hl[i]*(pcur[ i+1] - pcur[ i])
        # right boundary
        ψ_r[i] = b_K_x_hr[i] * ψ_r[i] + a_x_hr[i]*(pcur[ii+1] - pcur[ii])
    end
end

@views function update_p!(pold, pcur, pnew, halo, fact,
                          ψ_l = nothing, ψ_r = nothing,
                          ξ_l = nothing, ξ_r = nothing,
                          a_x_l = nothing, a_x_r = nothing,
                          b_K_x_l = nothing, b_K_x_r = nothing)
    nx, _... = size(pcur)
    for i = 2:nx-1
        d2p_dx2 = (pcur[i+1] - 2.0*pcur[i] + pcur[i-1])

        if i <= halo+1
            # left boundary
            dψ_dx = (ψ_l[i] - ψ_l[i-1])
            ξ_l[i-1] = b_K_x_l[i-1] * ξ_l[i-1] + a_x_l[i-1] * (d2p_dx2 + dψ_dx)
            damp = fact[i] * (dψ_dx + ξ_l[i-1])
        elseif i >= nx - halo
            # right boundary
            ii = i - (nx - halo) + 2
            dψ_dx = (ψ_r[ii] - ψ_r[ii-1])
            ξ_r[ii-1] = b_K_x_r[ii-1] * ξ_r[ii-1] + a_x_r[ii-1] * (d2p_dx2 + dψ_dx)
            damp = fact[i] * (dψ_dx + ξ_r[ii-1])
        else
            damp = 0.0
        end

        # update pressure
        pnew[i] = 2.0 * pcur[i] - pold[i] + fact[i] * (d2p_dx2) + damp
    end
end

@views function inject_sources!(pnew, dt2srctf, possrcs, it)
    _, nsrcs = size(dt2srctf)
    for i = 1:nsrcs
        isrc = possrcs[i,1]
        pnew[isrc] += dt2srctf[it,i]
    end
end

@views function kernel!(
    pold, pcur, pnew, fact,
    halo, ψ_l, ψ_r, ξ_l, ξ_r,
    a_x_hl, a_x_hr, b_K_x_hl, b_K_x_hr,
    a_x_l, a_x_r, b_K_x_l, b_K_x_r,
    possrcs, dt2srctf, it
)
    update_ψ!(ψ_l, ψ_r, pcur,
              halo,
              a_x_hl, a_x_hr,
              b_K_x_hl, b_K_x_hr)
    update_p!(pold, pcur, pnew, halo, fact,
              ψ_l, ψ_r,
              ξ_l, ξ_r,
              a_x_l, a_x_r,
              b_K_x_l, b_K_x_r)
    inject_sources!(pnew, dt2srctf, possrcs, it)

    return pcur, pnew, pold
end

@views function acoustic1D(
    dx::Float64,
    nt::Integer,
    vel::Vector{Float64};
    halo::Integer = 20,
    rcoef::Float64 = 0.0001,
    do_vis::Bool = true,
    do_bench::Bool = false,
    nvis::Integer = 5,
    gif_name::String = "acoustic1D",
    ylims::Vector{Float64} = [-4.0, 4.0]
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
    dt = dx / vel_max                    # timestep size (CFL + Courant condition) [s]
    times = collect(range(0.0,step=dt,length=nt))   # time vector [s]
    # CPML numerics
    alpha_max        = 2.0*π*(f0/2.0)
    thickness_cpml_x = halo * dx
    d0_x             = -(npower + 1) * vel_max * log(rcoef) / (2.0 * thickness_cpml_x)
    a_x_l , a_x_r , b_K_x_l , b_K_x_r  = calc_Kab_CPML(halo,dt,npower,d0_x,alpha_max,K_max,"ongrd")
    a_x_hl, a_x_hr, b_K_x_hl, b_K_x_hr = calc_Kab_CPML(halo,dt,npower,d0_x,alpha_max,K_max,"halfgrd")
    # precomputations
    fact = vel.^2 .* (dt^2/dx^2)

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
    possrcs = zeros(Int,3,1)
    possrcs[1,1] = div( nx, 4)
    possrcs[2,1] = div(2nx, 4)
    possrcs[3,1] = div(3nx, 4)
    # source time functions
    dt2srctf = zeros(nt,3)
    dt2srctf[:,1] .= fact[possrcs[1,1]] .* rickersource1D(times, t0, f0)
    dt2srctf[:,2] .= fact[possrcs[2,1]] .* rickersource1D(times, t0, f0)
    dt2srctf[:,3] .= fact[possrcs[3,1]] .* rickersource1D(times, t0, f0)

    # benchmarking instead of actual computation
    if do_bench
        t_it = @belapsed $kernel!(
            $pold, $pcur, $pnew, $fact,
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

    # time loop
    anim = Animation()
    for it=1:nt
        pold, pcur, pnew = kernel!(
            pold, pcur, pnew, fact,
            halo, ψ_l, ψ_r, ξ_l, ξ_r,
            a_x_hl, a_x_hr, b_K_x_hl, b_K_x_hr,
            a_x_l, a_x_r, b_K_x_l, b_K_x_r,
            possrcs, dt2srctf, it
        )

        # visualization
        if do_vis && (it % nvis == 0)
            # update ylims
            ylims[1] = min(ylims[1], minimum(pnew))
            ylims[2] = max(ylims[2], maximum(pnew))
            # sources positions
            scatter(possrcs[:,1], fill(0.0, size(possrcs, 1)); markershape=:star, color=:red, label="sources")
            # CPML boundaries
            plot!(fill(halo+1, 2), ylims; color=:grey, linestyle=:dot, label="CPML boundary")
            plot!(fill(nx-halo, 2), ylims; color=:grey, linestyle=:dot, label=:none)
            # plot pressure
            plot!(1:nx, pnew;
                 ylim=ylims, xlabel="lx", ylabel="pressure", label="pressure", color=:blue,
                 title="1D Acoustic CPML\n(halo=$(halo), rcoef=$(rcoef))")
            frame(anim)
        end
    end
    # save visualization
    if do_vis
        gif(anim, joinpath(DOCS_FLD, "$(gif_name).gif"))
    end

    return nothing
end

acoustic1D(10.0, 300, 2000 .* ones(Float64, 200); nvis=1, halo=20, rcoef=0.0001, do_vis=true)
# acoustic1D(2000 .* ones(Float64, 1_000_000), nt=1000, do_bench=true)