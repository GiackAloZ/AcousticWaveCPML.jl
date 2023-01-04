using Plots, Plots.Measures
using BenchmarkTools
using Printf

import ..AcousticWaveCPML: DOCS_FLD, calc_Kab_CPML, check_trial, Sources, Receivers

@views function update_ψ!(ψ_x_l, ψ_x_r, ψ_y_l, ψ_y_r, pcur,
                          halo, _dx, _dy,
                          a_x_hl, a_x_hr,
                          b_K_x_hl, b_K_x_hr,
                          a_y_hl, a_y_hr,
                          b_K_y_hl, b_K_y_hr)
    nx, ny = size(pcur)
    # x boundaries
    for j = 1:ny
        for i = 1:halo+1
            ii = i + nx - halo - 2  # shift for right boundary pressure indices
            # left boundary
            ψ_x_l[i,j] = b_K_x_hl[i] * ψ_x_l[i,j] + a_x_hl[i]*(pcur[ i+1,j] - pcur[ i,j])*_dx
            # right boundary
            ψ_x_r[i,j] = b_K_x_hr[i] * ψ_x_r[i,j] + a_x_hr[i]*(pcur[ii+1,j] - pcur[ii,j])*_dx
        end
    end
    # y boundaries
    for j = 1:halo+1
        for i = 1:nx
            jj = j + ny - halo - 2  # shift for bottom boundary pressure indices
            # top boundary
            ψ_y_l[i,j] = b_K_y_hl[j] * ψ_y_l[i,j] + a_y_hl[j]*(pcur[i, j+1] - pcur[i, j])*_dy
            # bottom boundary
            ψ_y_r[i,j] = b_K_y_hr[j] * ψ_y_r[i,j] + a_y_hr[j]*(pcur[i,jj+1] - pcur[i,jj])*_dy
        end
    end
end

@views function update_p!(pold, pcur, pnew, halo, fact, _dx, _dx2, _dy, _dy2,
                          ψ_x_l, ψ_x_r, ψ_y_l, ψ_y_r,
                          ξ_x_l, ξ_x_r, ξ_y_l, ξ_y_r,
                          a_x_l, a_x_r, b_K_x_l, b_K_x_r,
                          a_y_l, a_y_r, b_K_y_l, b_K_y_r)
    nx, ny = size(pcur)
    for j = 2:ny-1
        for i = 2:nx-1
            d2p_dx2 = (pcur[i+1,j] - 2.0*pcur[i,j] + pcur[i-1,j])*_dx2
            d2p_dy2 = (pcur[i,j+1] - 2.0*pcur[i,j] + pcur[i,j-1])*_dy2

            damp = 0.0
            # x boundaries
            if i <= halo+1
                # left boundary
                dψ_x_dx = (ψ_x_l[i,j] - ψ_x_l[i-1,j])*_dx
                ξ_x_l[i-1,j] = b_K_x_l[i-1] * ξ_x_l[i-1,j] + a_x_l[i-1] * (d2p_dx2 + dψ_x_dx)
                damp += fact[i,j] * (dψ_x_dx + ξ_x_l[i-1,j])
            elseif i >= nx - halo
                # right boundary
                ii = i - (nx - halo) + 2
                dψ_x_dx = (ψ_x_r[ii,j] - ψ_x_r[ii-1,j])*_dx
                ξ_x_r[ii-1,j] = b_K_x_r[ii-1] * ξ_x_r[ii-1,j] + a_x_r[ii-1] * (d2p_dx2 + dψ_x_dx)
                damp += fact[i,j] * (dψ_x_dx + ξ_x_r[ii-1,j])
            end
            # y boundaries
            if j <= halo+1
                # left boundary
                dψ_y_dy = (ψ_y_l[i,j] - ψ_y_l[i,j-1])*_dy
                ξ_y_l[i,j-1] = b_K_y_l[j-1] * ξ_y_l[i,j-1] + a_y_l[j-1] * (d2p_dy2 + dψ_y_dy)
                damp += fact[i,j] * (dψ_y_dy + ξ_y_l[i,j-1])
            elseif j >= ny - halo
                # right boundary
                jj = j - (ny - halo) + 2
                dψ_y_dy = (ψ_y_r[i,jj] - ψ_y_r[i,jj-1])*_dy
                ξ_y_r[i,jj-1] = b_K_y_r[jj-1] * ξ_y_r[i,jj-1] + a_y_r[jj-1] * (d2p_dy2 + dψ_y_dy)
                damp += fact[i,j] * (dψ_y_dy + ξ_y_r[i,jj-1])
            end

            # update pressure
            pnew[i,j] = 2.0 * pcur[i,j] - pold[i,j] + fact[i,j] * (d2p_dx2 + d2p_dy2) + damp
        end
    end
end

@views function inject_sources!(pnew, dt2srctf, possrcs, it)
    _, nsrcs = size(dt2srctf)
    for s = 1:nsrcs
        isrc = possrcs[s,1]
        jsrc = possrcs[s,2]
        pnew[isrc,jsrc] += dt2srctf[it,s]
    end
end

@views function record_receivers!(pnew, traces, posrecs, it)
    _, nrecs = size(traces)
    for s = 1:nrecs
        irec = posrecs[s,1]
        jrec = posrecs[s,2]
        traces[it,s] = pnew[irec, jrec]
    end
end

@views function forward!(
    pold, pcur, pnew, fact, _dx, _dx2, _dy, _dy2,
    halo, ψ_x_l, ψ_x_r, ξ_x_l, ξ_x_r, ψ_y_l, ψ_y_r, ξ_y_l, ξ_y_r,
    a_x_hl, a_x_hr, b_K_x_hl, b_K_x_hr,
    a_x_l, a_x_r, b_K_x_l, b_K_x_r,
    a_y_hl, a_y_hr, b_K_y_hl, b_K_y_hr,
    a_y_l, a_y_r, b_K_y_l, b_K_y_r,
    possrcs, dt2srctf,
    posrecs, traces,
    it
)
    update_ψ!(ψ_x_l, ψ_x_r, ψ_y_l, ψ_y_r, pcur,
              halo, _dx, _dy,
              a_x_hl, a_x_hr,
              b_K_x_hl, b_K_x_hr,
              a_y_hl, a_y_hr,
              b_K_y_hl, b_K_y_hr)
    update_p!(pold, pcur, pnew, halo, fact, _dx, _dx2, _dy, _dy2,
              ψ_x_l, ψ_x_r, ψ_y_l, ψ_y_r,
              ξ_x_l, ξ_x_r, ξ_y_l, ξ_y_r,
              a_x_l, a_x_r, b_K_x_l, b_K_x_r,
              a_y_l, a_y_r, b_K_y_l, b_K_y_r)
    inject_sources!(pnew, dt2srctf, possrcs, it)
    record_receivers!(pnew, traces, posrecs, it)

    return pcur, pnew, pold
end

@views function solve2D(
    lx::Real,
    ly::Real,
    lt::Real,
    vel::Matrix{<:Real},
    srcs::Sources,
    recs::Receivers;
    halo::Integer = 20,
    rcoef::Real = 0.0001,
    ppw::Real = 10.0,
    freetop::Bool = true,
    do_bench::Bool = false,
    do_vis::Bool = false,
    nvis::Integer = 5,
    gif_name::String = "acoustic2D",
    plims::Vector{<:Real} = [-1.0, 1.0],
    threshold::Real = 0.01
)
    ###################################################
    # MODEL SETUP
    ###################################################
    # Physics
    f0 = srcs.freqdomain                            # dominating frequency [Hz]
    # Derived physics
    vel_max = maximum(vel)                          # maximum velocity [m/s]
    # Numerics
    nx, ny = size(vel)                              # number of grid points
    # Derived numerics
    dx = lx / (nx-1)                                # grid step size [m]
    dy = ly / (ny-1)                                # grid step size [m]
    dt = sqrt(2) / (vel_max * (1/dx + 1/dy))/2      # maximum possible timestep size (CFL stability condition) [s]
    nt = ceil(Int, lt / dt)                         # number of timesteps
    times = collect(range(0.0,step=dt,length=nt))   # time vector [s]
    # CPML numerics
    alpha_max     = π*f0                            # CPML α multiplicative factor (half of dominating angular frequency)
    npower        = 2.0                             # CPML power coefficient
    K_max         = 1.0                             # CPML K coefficient value
    thickness_cpml_x = halo * dx                    # CPML x-direction layer thickness [m]
    thickness_cpml_y = halo * dy                    # CPML y-direction layer thickness [m]
    d0_x          = -(npower + 1) * vel_max * log(rcoef) / (2.0 * thickness_cpml_x)     # x-direction damping profile
    d0_y          = -(npower + 1) * vel_max * log(rcoef) / (2.0 * thickness_cpml_y)     # y-direction damping profile
    # CPML coefficients (l = left, r = right, h = staggered in betweeen grid points)
    a_x_l , a_x_r , b_K_x_l , b_K_x_r  = calc_Kab_CPML(halo,dt,npower,d0_x,alpha_max,K_max,"ongrd")
    a_x_hl, a_x_hr, b_K_x_hl, b_K_x_hr = calc_Kab_CPML(halo,dt,npower,d0_x,alpha_max,K_max,"halfgrd")
    a_y_l , a_y_r , b_K_y_l , b_K_y_r  = calc_Kab_CPML(halo,dt,npower,d0_y,alpha_max,K_max,"ongrd")
    a_y_hl, a_y_hr, b_K_y_hl, b_K_y_hr = calc_Kab_CPML(halo,dt,npower,d0_y,alpha_max,K_max,"halfgrd")
    # Free top BDC (no CPML in top boundary)
    if freetop
        a_y_l .= 0.0
        a_y_hl .= 0.0
        b_K_y_l .= 1.0
        b_K_y_hl .= 1.0
    end
    ###################################################

    ###################################################
    # PRECOMPUTATIONS
    ###################################################
    _dx = 1.0 / dx
    _dx2 = 1.0 / (dx^2)
    _dy = 1.0 / dy
    _dy2 = 1.0 / (dy^2)
    fact = (dt^2) .* (vel .^ 2)
    ###################################################

    ###################################################
    # ASSERTIONS
    ###################################################
    @assert sqrt(dx^2 + dy^2) <= vel_max/(ppw * f0) "Not enough points per wavelength!"
    ###################################################
    
    ###################################################
    # ARRAYS INITIALIZATION
    ###################################################
    # pressure arrays
    pold = zeros(nx,ny)                                 # old pressure     (it-1) [Pas]
    pcur = zeros(nx,ny)                                 # current pressure (it)   [Pas]
    pnew = zeros(nx,ny)                                 # next pressure    (it+1) [Pas]
    # CPML arrays
    ψ_x_l, ψ_x_r = zeros(halo+1,ny), zeros(halo+1,ny)   # left and right ψ in x-boundary
    ξ_x_l, ξ_x_r = zeros(halo,ny), zeros(halo,ny)       # left and right ξ in x-boundary
    ψ_y_l, ψ_y_r = zeros(nx,halo+1), zeros(nx,halo+1)   # top and bottom ψ in y-boundary
    ξ_y_l, ξ_y_r = zeros(nx,halo), zeros(nx,halo)       # top and bottom ξ in y-boundary
    ###################################################

    ###################################################
    # SOURCES / RECEIVERS SETUP
    ###################################################
    # source time functions
    nsrcs = srcs.n                                      # number of sources
    dt2srctf = zeros(nt, nsrcs)                         # scaled source time functions (prescaling with boxcar function 1/(dx*dy))
    for s = 1:nsrcs
        dt2srctf[:,s] .= (dt^2 / (dx*dy)) .* srcs.srctfs[s].(times, srcs.t0s[s], f0)
    end
    # find nearest grid point for each source
    possrcs = zeros(Int, size(srcs.positions))          # sources positions (in grid points)
    for s = 1:nsrcs
        possrcs[s,:] .= round.(Int, [srcs.positions[s,1] / dx + 1, srcs.positions[s,2] / dy + 1], RoundNearestTiesUp)
    end
    @assert all(1 .<= possrcs[:,1] .<= nx) && all(1 .<= possrcs[:,2] .<= ny) "At least one source is not inside the model!"
    nrecs = recs.n                                      # number of receivers
    traces = zeros(nt, nrecs)                           # receiver seismograms
    # find nearest grid point for each receiver
    posrecs = zeros(Int, size(recs.positions))          # receiver positions (in grid points)
    for r = 1:nrecs
        posrecs[r,:] .= round.(Int, [recs.positions[r,1] / dx + 1, recs.positions[r,2] / dy + 1], RoundNearestTiesUp)
    end
    @assert all(1 .<= posrecs[:,1] .<= nx) && all(1 .<= posrecs[:,2] .<= ny) "At least one receiver is not inside the model!"
    ###################################################

    ###################################################
    # BENCHMARKING (with BenchmarkTools)
    ###################################################
    if do_bench
        # run benchmark trial
        trial = @benchmark $forward!(
            $pold,    $pcur,    $pnew,      $fact, $_dx, $_dx2, $_dy, $_dy2,
            $halo,    $ψ_x_l,   $ψ_x_r,     $ξ_x_l, $ξ_x_r, $ψ_y_l, $ψ_y_r, $ξ_y_l, $ξ_y_r,
            $a_x_hl,  $a_x_hr,  $b_K_x_hl,  $b_K_x_hr,
            $a_x_l,   $a_x_r,   $b_K_x_l,   $b_K_x_r,
            $a_y_hl,  $a_y_hr,  $b_K_y_hl,  $b_K_y_hr,
            $a_y_l,   $a_y_r,   $b_K_y_l,   $b_K_y_r,
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
                     3*(nx*ny) +
                     2*((halo+1)*ny) + 2*(halo*ny) +
                     2*((halo+1)*nx) + 2*(halo*nx) +
                     4*2*(halo+1) + 4*2*halo
                    ) * sizeof(Float64) / 1e9
        # effective memory access [GB]
        A_eff = (
            (halo+1)*ny*2*(2 + 1) +         # update_ψ_x!
            (halo+1)*nx*2*(2 + 1) +         # update_ψ_y!
            (halo+1)*ny*2*(1 + 1) +         # update ξ_x in update_p!
            (halo+1)*nx*2*(1 + 1) +         # update ξ_y in update_p!
            4*nx*ny                         # update_p! (inner points)
        ) * sizeof(Float64) / 1e9
        # effective memory throughput [GB/s]
        T_eff = A_eff / t_it
        @printf("size = %dx%d, time = %1.3e sec, Teff = %1.3f GB/s, memory = %1.3f GB\n", nx, ny, t_it, T_eff, alloc_mem)
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
        default(size=(1400, 1400), framestyle=:box, grid=false, margin=20pt, legendfontsize=14, labelfontsize=14)
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
            pold, pcur, pnew, fact, _dx, _dx2, _dy, _dy2,
            halo, ψ_x_l, ψ_x_r, ξ_x_l, ξ_x_r, ψ_y_l, ψ_y_r, ξ_y_l, ξ_y_r,
            a_x_hl, a_x_hr, b_K_x_hl, b_K_x_hr,
            a_x_l, a_x_r, b_K_x_l, b_K_x_r,
            a_y_hl, a_y_hr, b_K_y_hl, b_K_y_hr,
            a_y_l, a_y_r, b_K_y_l, b_K_y_r,
            possrcs, dt2srctf, posrecs, traces, it
        )

        # visualization
        if do_vis && (it % nvis == 0)
            # velocity model heatmap
            velview = (((copy(vel) .- minimum(vel)) ./ (maximum(vel) - minimum(vel)))) .* (plims[2] - plims[1]) .+ plims[1]
            p1 = heatmap(0:dx:lx, 0:dy:ly, velview'; c=:grayC, aspect_ratio=:equal, colorbar=false)
            # pressure heatmap
            pview = copy(pcur)
            # print iteration values
            maxabsp = @sprintf "%e" maximum(abs.(pview))
            @show it*dt, it, maxabsp
            # threshold values
            pview[(pview .> plims[1] * threshold) .& (pview .< plims[2] * threshold)] .= NaN
            # heatmap
            heatmap!(0:dx:lx, 0:dy:ly, pview';
                  xlims=(0,lx),ylims=(0,ly), clims=(plims[1], plims[2]), aspect_ratio=:equal,
                  xlabel="lx [m]", ylabel="ly [m]", clabel="pressure", c=:diverging_bwr_20_95_c54_n256, colorbar=false,
                  title="Pressure [2D Acoustic CPML]\n(nx=$(nx), ny=$(ny), halo=$(halo), rcoef=$(rcoef), threshold=$(round(threshold * 100, digits=2))%)\nit=$(it), time=$(round(it*dt, digits=2)) [sec], maxabsp=$(maxabsp) [Pas]"
            )
            # sources positions
            scatter!((possrcs[:,1].-1) .* dx, (possrcs[:,2].-1) .* dy; markersize=10, markerstrokewidth=0, markershape=:star, color=:red, label="sources")
            # receivers positions
            scatter!((posrecs[:,1].-1) .* dx, (posrecs[:,2].-1) .* dy; markersize=10, markerstrokewidth=0, markershape=:dtriangle, color=:blue, label="receivers")
            
            # CPML boundaries
            if freetop
                plot!(fill(halo * dx, 2), [0, ly - (halo * dy)]; lw=2, color=:grey, linestyle=:dot, label="CPML boundary")
                plot!(fill(lx - (halo * dx), 2), [0, ly - (halo * dy)]; lw=2, color=:grey, linestyle=:dot, label=:none)
                plot!([halo * dx, lx - (halo * dx)], fill(ly - (halo * dy), 2); lw=2, color=:grey, linestyle=:dot, label=:none)
            else
                plot!(fill(halo * dx, 2), [halo * dy, ly - (halo * dy)]; lw=2, color=:grey, linestyle=:dot, label="CPML boundary")
                plot!(fill(lx - (halo * dx), 2), [halo * dy, ly - (halo * dy)]; lw=2, color=:grey, linestyle=:dot, label=:none)
                plot!([halo * dx, lx - (halo * dx)], fill(ly - (halo * dy), 2); lw=2, color=:grey, linestyle=:dot, label=:none)
                plot!([halo * dx, lx - (halo * dx)], fill(halo * dy, 2); lw=2, color=:grey, linestyle=:dot, label=:none)
            end
            # flip y axis
            yflip!(true)

            # traces plot
            p2 = plot(times[1:it], traces[1:it, :];
                ylims=plims,
                xlims=(times[1], times[end]),
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
