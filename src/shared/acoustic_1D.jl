using Plots, Plots.Measures
using BenchmarkTools
using Printf

import ..AcousticWaveCPML: DOCS_FLD, calc_Kab_CPML, Sources, Receivers

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

@views function update_p!(pold, pcur, pnew, halo, fact, _dx, _dx2,
                          ψ_l, ψ_r,
                          ξ_l, ξ_r,
                          a_x_l, a_x_r,
                          b_K_x_l, b_K_x_r)
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

@views function inject_sources!(pnew, dt2srctf, possrcs, it)
    _, nsrcs = size(dt2srctf)
    for i = 1:nsrcs
        isrc = possrcs[i,1]
        pnew[isrc] += dt2srctf[it,i]
    end
end

@views function record_receivers!(pnew, traces, posrecs, it)
    _, nrecs = size(traces)
    for s = 1:nrecs
        irec = posrecs[s,1]
        traces[it,s] = pnew[irec]
    end
end

# Kernel 
@views function forward!(
    pold, pcur, pnew, fact, _dx, _dx2,
    halo, ψ_l, ψ_r, ξ_l, ξ_r,
    a_x_hl, a_x_hr, b_K_x_hl, b_K_x_hr,
    a_x_l, a_x_r, b_K_x_l, b_K_x_r,
    possrcs, dt2srctf, posrecs, traces, it
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
    record_receivers!(pnew, traces, posrecs, it)

    return pcur, pnew, pold
end

@views function solve1D(
    lx::Real,
    lt::Real,
    vel::Vector{<:Real},
    srcs::Sources,
    recs::Receivers;
    halo::Integer = 20,
    rcoef::Real = 0.0001,
    ppw::Real = 10.0,
    do_bench::Bool = false,
    do_vis::Bool = false,
    nvis::Integer = 5,
    gif_name::String = "acoustic1D",
    plims::Vector{<:Real} = [-1.0, 1.0]
)
    ###################################################
    # MODEL SETUP
    ###################################################
    # Physics
    f0 = srcs.freqdomain                            # dominating frequency [Hz]
    # Derived physics
    vel_max = maximum(vel)                          # maximum velocity [m/s]
    # Numerics
    nx            = length(vel)                     # number of grid points
    # Derived numerics
    dx = lx / (nx-1)                                # grid step size [m]
    dt = dx / vel_max                               # maximum possible timestep size (CFL stability condition) [s]
    nt = ceil(Int, lt / dt)                         # number of timesteps
    times = collect(range(0.0,step=dt,length=nt))   # time vector [s]
    # CPML numerics
    alpha_max     = π*f0                            # CPML α multiplicative factor (half of dominating angular frequency)
    npower        = 2.0                             # CPML power coefficient
    K_max         = 1.0                             # CPML K coefficient value
    thickness_cpml_x = halo * dx                    # CPML x-direction layer thickness [m]
    d0_x          = -(npower + 1) * vel_max * log(rcoef) / (2.0 * thickness_cpml_x)     # x-direction damping profile
    # CPML coefficients (l = left, r = right, h = staggered in betweeen grid points)
    a_x_l , a_x_r , b_K_x_l , b_K_x_r  = calc_Kab_CPML(halo,dt,npower,d0_x,alpha_max,K_max,"ongrd")
    a_x_hl, a_x_hr, b_K_x_hl, b_K_x_hr = calc_Kab_CPML(halo,dt,npower,d0_x,alpha_max,K_max,"halfgrd")
    ###################################################

    ###################################################
    # PRECOMPUTATIONS
    ###################################################
    _dx = 1.0 / dx
    _dx2 = 1.0 / (dx^2)
    fact = (dt^2) .* (vel .^ 2)
    ###################################################

    ###################################################
    # ASSERTIONS
    ###################################################
    @assert dx <= vel_max/(ppw * f0) "Not enough points per wavelength!"
    ###################################################
    
     ###################################################
    # ARRAYS INITIALIZATION
    ###################################################
    # pressure arrays
    pold = zeros(nx)                                # old pressure     (it-1) [Pas]
    pcur = zeros(nx)                                # current pressure (it)   [Pas]
    pnew = zeros(nx)                                # next pressure    (it+1) [Pas]
    # CPML arrays
    ψ_l, ψ_r = zeros(halo+1), zeros(halo+1)     # left and right ψ in x-boundary
    ξ_l, ξ_r = zeros(halo), zeros(halo)         # left and right ξ in x-boundary
    ###################################################

    ###################################################
    # SOURCES / RECEIVERS SETUP
    ###################################################
    # source time functions
    nsrcs = srcs.n                                      # number of sources
    dt2srctf = zeros(nt, nsrcs)                         # scaled source time functions (prescaling with boxcar function 1/dx)
    for s = 1:nsrcs
        dt2srctf[:,s] .= (dt^2 / dx) .* srcs.srctfs[s].(times, srcs.t0s[s], f0)
    end
    # find nearest grid point for each source
    possrcs = zeros(Int, size(srcs.positions))          # sources positions (in grid points)
    for s = 1:nsrcs
        possrcs[s,:] .= round.(Int, [srcs.positions[s,1] / dx + 1], RoundNearestTiesUp)
    end
    @assert all(1 .<= possrcs[:,1] .<= nx) "At least one source is not inside the model!"
    nrecs = recs.n                                      # number of receivers
    traces = zeros(nt, nrecs)                           # receiver seismograms
    # find nearest grid point for each receiver
    posrecs = zeros(Int, size(recs.positions))          # receiver positions (in grid points)
    for r = 1:nrecs
        posrecs[r,:] .= round.(Int, [recs.positions[r,1] / dx + 1], RoundNearestTiesUp)
    end
    @assert all(1 .<= posrecs[:,1] .<= nx) "At least one receiver is not inside the model!"
    ###################################################

    ###################################################
    # BENCHMARKING (with BenchmarkTools)
    ###################################################
    if do_bench
        trial = @benchmark $forward!(
            $pold, $pcur, $pnew, $fact, $_dx, $_dx2,
            $halo, $ψ_l, $ψ_r, $ξ_l, $ξ_r,
            $a_x_hl, $a_x_hr, $b_K_x_hl, $b_K_x_hr,
            $a_x_l, $a_x_r, $b_K_x_l, $b_K_x_r,
            $possrcs, $dt2srctf, $posrecs, $traces, 1
        )
        # check benchmark
        confidence = 0.95
        med_range = 0.05
        pass, ci, tol_range, t_it_mean = check_trial(trial, confidence, med_range)
        t_it = minimum(trial).time / 1e9
        if !pass
            @printf("Statistical trial check not passed!\nmedian = %g [sec]\n%d%% tolerance range = (%g, %g) [sec]\n%d%% CI = (%g, %g) [sec]\n", t_it_mean, med_range*100, tol_range[1], tol_range[2], confidence*100, ci[1], ci[2])
        end
        # allocated memory [GB]
        alloc_mem = (
                     3*nx +
                     2*(halo+1) + 2*halo +
                     4*(halo+1) + 4*halo
                    ) * sizeof(Float64) / 1e9
        # effective memory access [GB]
        A_eff = (
            2*( 4*(halo+1) + (halo+1) ) +                   # update_ψ!
            2*( 4*(halo+1) + (halo+1) ) +                   # update_p! (CPML layers)
            (4*nx + nx)                                     # update_p! (all points)
        ) * sizeof(Float64) / 1e9
        # effective memory throughput [GB/s]
        T_eff = A_eff / t_it
        @printf("size = %d, time = %1.3e sec, Teff = %1.3f GB/s, memory = %1.3f GB\n", nx, t_it, T_eff, alloc_mem)
        return nothing
    end
    ###################################################

    ###################################################
    # VISUALIZATION SETUP
    ###################################################
    if do_vis
        # Disable interactive visualization
        ENV["GKSwstype"]="nul"
        # Set default plot values
        default(size=(1400, 1400), framestyle=:box, grid=false, margin=20pt, legendfontsize=14, thickness_scaling=1)
        # Create results folders if not present
        mkpath(DOCS_FLD)
        # Create animation object
        anim = Animation()
    end
    ###################################################

    ###################################################
    # TIME LOOP
    ###################################################
    for it=1:nt
        # compute single forward time step
        pold, pcur, pnew = forward!(
            pold, pcur, pnew, fact, _dx, _dx2,
            halo, ψ_l, ψ_r, ξ_l, ξ_r,
            a_x_hl, a_x_hr, b_K_x_hl, b_K_x_hr,
            a_x_l, a_x_r, b_K_x_l, b_K_x_r,
            possrcs, dt2srctf, posrecs, traces, it
        )

        # visualization
        if do_vis && (it % nvis == 0)
            # update plims
            plims[1] = min(plims[1], minimum(pcur))
            plims[2] = max(plims[2], maximum(pcur))
            # print iteration values
            maxabsp = @sprintf "%e" maximum(abs.(pcur))
            @show it*dt, it, maxabsp, plims
            # plot pressure
            p1 = plot(0:dx:lx, pcur;
                  ylim=plims, xlims=(0,lx), xlabel="lx", ylabel="pressure", label="pressure", color=:blue,
                  title="Pressure [1D Acoustic CPML]\n(nx=$(nx), halo=$(halo), rcoef=$(rcoef))\nit=$(it), time=$(round(it*dt, digits=2)) [sec], maxabsp=$(maxabsp) [Pas]")
            # sources positions
            scatter!((possrcs[:,1] .- 1) .* dx, fill(0.0, size(possrcs, 1)); markersize=10, markerstrokewidth=0, markershape=:star, color=:red, label="sources")
            # receivers positions
            scatter!((posrecs[:,1] .- 1) .* dx, fill(0.0, size(posrecs, 1)); markersize=10, markerstrokewidth=0, markershape=:dtriangle, color=:blue, label="receivers")
            # CPML boundaries
            plot!(fill(halo * dx, 2), plims; color=:grey, linestyle=:dot, label="CPML boundary")
            plot!(fill(lx - (halo * dx), 2), plims; color=:grey, linestyle=:dot, label=:none)
            
            # traces plot
            p2 = plot(times[1:it] .+ dt, traces[1:it, :];
                ylims=plims,
                xlims=(0, lt),
                xlabel="time [sec]",
                ylabel="pressure [Pas]",
                title="Receivers seismograms",
                labels=reshape(["receiver $(i)" for i in 1:nrecs], (1,nrecs))
            )

            # layout
            l = @layout [
                a{0.7h}
                b
            ]
            plot(p1, p2, layout=l)

            # save frame
            frame(anim)
        end
    end
    ###################################################

    ###################################################
    # SAVE RESULTS
    ###################################################
    if do_vis
        gif(anim, joinpath(DOCS_FLD, "$(gif_name).gif"); fps=5)
    end
    # save seismograms traces
    recs.seismograms = copy(traces)
    ###################################################

    return pcur
end
