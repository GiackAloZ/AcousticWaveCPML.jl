using Plots, Plots.Measures
using BenchmarkTools
using Printf

import Logging
Logging.disable_logging(Logging.Warn)

default(size=(1000, 600), framestyle=:box,grid=false,margin=20pt)

include("utils.jl")

DOCS_FLD = joinpath(dirname(@__DIR__), "docs")

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
            ψ_x_l[i,j] = b_K_x_hl[i] * ψ_x_l[i] + a_x_hl[i]*(pcur[ i+1,j] - pcur[ i,j])*_dx
            # right boundary
            ψ_x_r[i,j] = b_K_x_hr[i] * ψ_x_r[i] + a_x_hr[i]*(pcur[ii+1,j] - pcur[ii,j])*_dx
        end
    end
    # y boundaries
    for j = 1:halo+1
        for i = 1:nx
            jj = j + ny - halo - 2  # shift for bottom boundary pressure indices
            # top boundary
            ψ_y_l[i,j] = b_K_y_hl[j] * ψ_y_l[j] + a_y_hl[j]*(pcur[i, j+1] - pcur[i, j])*_dy
            # bottom boundary
            ψ_y_r[i,j] = b_K_y_hr[j] * ψ_y_r[j] + a_y_hr[j]*(pcur[i,jj+1] - pcur[i,jj])*_dy
        end
    end
end

@views function update_p!(pold, pcur, pnew, halo, fact, _dx, _dx2, _dy, _dy2,
                          ψ_x_l = nothing, ψ_x_r = nothing, ψ_y_l = nothing, ψ_y_r = nothing,
                          ξ_x_l = nothing, ξ_x_r = nothing, ξ_y_l = nothing, ξ_y_r = nothing,
                          a_x_l = nothing, a_x_r = nothing, b_K_x_l = nothing, b_K_x_r = nothing,
                          a_y_l = nothing, a_y_r = nothing, b_K_y_l = nothing, b_K_y_r = nothing)
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

@views function kernel!(
    pold, pcur, pnew, fact, _dx, _dx2, _dy, _dy2,
    halo, ψ_x_l, ψ_x_r, ξ_x_l, ξ_x_r, ψ_y_l, ψ_y_r, ξ_y_l, ξ_y_r,
    a_x_hl, a_x_hr, b_K_x_hl, b_K_x_hr,
    a_x_l, a_x_r, b_K_x_l, b_K_x_r,
    a_y_hl, a_y_hr, b_K_y_hl, b_K_y_hr,
    a_y_l, a_y_r, b_K_y_l, b_K_y_r,
    possrcs, dt2srctf, it
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

    return pcur, pnew, pold
end

@views function acoustic2D(
    lx::Real,
    ly::Real,
    nt::Integer,
    vel::Matrix{<:Real},
    possrcs;
    halo::Integer = 20,
    rcoef::Real = 0.0001,
    do_vis::Bool = true,
    do_bench::Bool = false,
    nvis::Integer = 5,
    gif_name::String = "acoustic2D",
    plims::Vector{<:Real} = [-3, 3],
    threshold::Real = 0.05,
    freetop::Bool = true
)
    # Physics
    f0 = 8.0                            # dominating frequency [Hz]
    t0 = 1.2 / f0                       # activation time [s]
    # Derived physics
    vel_max = maximum(vel)              # maximum velocity [m/s]
    # Numerics
    nx, ny        = size(vel)         # number of grid points
    npower        = 2.0
    K_max         = 1.0
    # Derived numerics
    dx = lx / (nx-1)                    # grid step size [m]
    dy = ly / (ny-1)                    # grid step size [m]
    dt = 0.0012                         # 1.0 / (sqrt(1.0/(dx^2) + 1.0/(dy^2))) / vel_max  # timestep size (CFL + Courant condition) [s]
    times = collect(range(0.0,step=dt,length=nt))   # time vector [s]
    # CPML numerics
    alpha_max        = 2.0*π*(f0/2.0)
    thickness_cpml_x = halo * dx
    thickness_cpml_y = halo * dy
    d0_x             = -(npower + 1) * vel_max * log(rcoef) / (2.0 * thickness_cpml_x)
    d0_y             = -(npower + 1) * vel_max * log(rcoef) / (2.0 * thickness_cpml_y)
    a_x_l , a_x_r , b_K_x_l , b_K_x_r  = calc_Kab_CPML(halo,dt,npower,d0_x,alpha_max,K_max,"ongrd")
    a_x_hl, a_x_hr, b_K_x_hl, b_K_x_hr = calc_Kab_CPML(halo,dt,npower,d0_x,alpha_max,K_max,"halfgrd")
    a_y_l , a_y_r , b_K_y_l , b_K_y_r  = calc_Kab_CPML(halo,dt,npower,d0_y,alpha_max,K_max,"ongrd")
    a_y_hl, a_y_hr, b_K_y_hl, b_K_y_hr = calc_Kab_CPML(halo,dt,npower,d0_y,alpha_max,K_max,"halfgrd")
    # free top boundary
    if freetop
        a_y_l .= 0.0
        a_y_hl .= 0.0
        b_K_y_l .= 1.0
        b_K_y_hl .= 1.0
    end

    # precomputations
    _dx = 1.0 / dx
    _dx2 = 1.0 / (dx^2)
    _dy = 1.0 / dy
    _dy2 = 1.0 / (dy^2)
    fact = (dt^2) .* vel.^2

    # assertions for stability
    @assert max(dx, dy) <= vel_max/(10.0 * f0)   # at least 10pts per wavelength
    
    # Array initialization

    # pressure arrays
    pold = zeros(nx,ny)
    pcur = zeros(nx,ny)
    pnew = zeros(nx,ny)
    # CPML arrays
    ψ_x_l, ψ_x_r = zeros(halo+1,ny), zeros(halo+1,ny)
    ξ_x_l, ξ_x_r = zeros(halo,ny), zeros(halo,ny)
    ψ_y_l, ψ_y_r = zeros(nx,halo+1), zeros(nx,halo+1)
    ξ_y_l, ξ_y_r = zeros(nx,halo), zeros(nx,halo)
    # source time functions
    nsrcs = size(possrcs,1)
    dt2srctf = zeros(nt,nsrcs)
    for s = 1:nsrcs
        dt2srctf[:,s] .= (dt^2) .* 1000 .* rickersource1D(times, t0, f0)
    end

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

    # time loop
    anim = Animation()
    for it=1:nt
        pold, pcur, pnew = kernel!(
            pold, pcur, pnew, fact, _dx, _dx2, _dy, _dy2,
            halo, ψ_x_l, ψ_x_r, ξ_x_l, ξ_x_r, ψ_y_l, ψ_y_r, ξ_y_l, ξ_y_r,
            a_x_hl, a_x_hr, b_K_x_hl, b_K_x_hr,
            a_x_l, a_x_r, b_K_x_l, b_K_x_r,
            a_y_hl, a_y_hr, b_K_y_hl, b_K_y_hr,
            a_y_l, a_y_r, b_K_y_l, b_K_y_r,
            possrcs, dt2srctf, it
        )

        # visualization
        if do_vis && (it % nvis == 0)
            # velocity model heatmap
            velview = (((copy(vel) .- minimum(vel)) ./ (maximum(vel) - minimum(vel)))) .* (plims[2] - plims[1]) .+ plims[1]
            heatmap(0:dx:lx, 0:dy:ly, velview'; c=:grayC, aspect_ratio=:equal)
            # pressure heatmap
            pview = copy(pcur) .* 1e3
            maxabsp = @sprintf "%e" maximum(abs.(pview))
            @show maxabsp
            pview[(pview .> plims[1] * threshold) .& (pview .< plims[2] * threshold)] .= NaN
            heatmap!(0:dx:lx, 0:dy:ly, pview';
                  xlims=(0,lx),ylims=(0,ly), clims=(plims[1], plims[2]), aspect_ratio=:equal,
                  xlabel="lx", ylabel="ly", clabel="pressure", c=:diverging_bwr_20_95_c54_n256,
                  title="2D Acoustic CPML\n(halo=$(halo), rcoef=$(rcoef), threshold=$(round(threshold * 100))%)\n max abs pressure = $(maxabsp)"
            )
            # sources positions
            scatter!((possrcs[:,1].-1) .* dx, (possrcs[:,2].-1) .* dy; markershape=:star, markersize=5, color=:red, label="sources")
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
            # save frame
            frame(anim)
        end
    end
    # save visualization
    if do_vis
        gif(anim, joinpath(DOCS_FLD, "$(gif_name).gif"))
    end

    return nothing
end

# gradient velocity model
nx, ny = 211, 121
vel = zeros(Float64, nx, ny);
for i=1:nx
    for j=1:ny
        vel[i,j] = 2000.0 + 12.0*(j-1)
    end
end
# constant after some depth
bigv = vel[1,ny-40]
vel[:,ny-40:end] .= bigv

# 6 equidistant sources on top
possrcs = zeros(Int,6,2)
possrcs[1,:] = [div(3nx, 11, RoundUp), 3]
possrcs[2,:] = [div(4nx, 11, RoundUp), 3]
possrcs[3,:] = [div(5nx, 11, RoundUp), 3]
possrcs[4,:] = [div(6nx, 11, RoundUp), 3]
possrcs[5,:] = [div(7nx, 11, RoundUp), 3]
possrcs[6,:] = [div(8nx, 11, RoundUp), 3]

acoustic2D(2100.0, 1200.0, 1500, vel, possrcs; halo=5, rcoef=0.01, do_vis=true, gif_name="acoustic2D_halo5")
acoustic2D(2100.0, 1200.0, 1500, vel, possrcs; halo=10, rcoef=0.001, do_vis=true, gif_name="acoustic2D_halo10")
acoustic2D(2100.0, 1200.0, 1500, vel, possrcs; halo=20, rcoef=0.0001, do_vis=true, gif_name="acoustic2D_halo20")
acoustic2D(2100.0, 1200.0, 1500, vel, possrcs; halo=40, rcoef=0.00001, do_vis=true, gif_name="acoustic2D_halo40")

# simple constant velocity model
nx, ny = 211, 211
vel = 2000.0 .* ones(Float64, nx, ny);
# one source in the center
possrcs = zeros(Int,1,2)
possrcs[1,:] = [div(nx, 2, RoundUp), div(ny, 2, RoundUp)]

acoustic2D(2100.0, 2100.0, 1000, vel, possrcs; halo=5, rcoef=0.01, do_vis=true, gif_name="acoustic2D_center_halo5", freetop=false, threshold=0.0)
acoustic2D(2100.0, 2100.0, 1000, vel, possrcs; halo=10, rcoef=0.001, do_vis=true, gif_name="acoustic2D_center_halo10", freetop=false, threshold=0.0)
acoustic2D(2100.0, 2100.0, 1000, vel, possrcs; halo=20, rcoef=0.0001, do_vis=true, gif_name="acoustic2D_center_halo20", freetop=false, threshold=0.0)
acoustic2D(2100.0, 2100.0, 1000, vel, possrcs; halo=40, rcoef=0.00001, do_vis=true, gif_name="acoustic2D_center_halo40", freetop=false, threshold=0.0)
