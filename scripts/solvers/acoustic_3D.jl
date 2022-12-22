using Plots, Plots.Measures
using BenchmarkTools
using Printf
using HDF5

default(size=(1000, 600), framestyle=:box,grid=false,margin=20pt)

include("../utils.jl")

# folders for results
DOCS_FLD = joinpath(dirname(dirname(@__DIR__)), "simulations")

# Disable interactive visualization
ENV["GKSwstype"]="nul"

@views function update_ψ!(ψ_x_l, ψ_x_r, ψ_y_l, ψ_y_r, ψ_z_l, ψ_z_r, pcur,
                          halo, _dx, _dy, _dz,
                          a_x_hl, a_x_hr,
                          b_K_x_hl, b_K_x_hr,
                          a_y_hl, a_y_hr,
                          b_K_y_hl, b_K_y_hr,
                          a_z_hl, a_z_hr,
                          b_K_z_hl, b_K_z_hr)
    nx, ny, nz = size(pcur)
    # x boundaries
    for k = 1:nz
        for j = 1:ny
            for i = 1:halo+1
                ii = i + nx - halo - 2  # shift for right boundary pressure indices
                # left boundary
                ψ_x_l[i,j,k] = b_K_x_hl[i] * ψ_x_l[i,j,k] + a_x_hl[i]*(pcur[ i+1,j,k] - pcur[ i,j,k])*_dx
                # right boundary
                ψ_x_r[i,j,k] = b_K_x_hr[i] * ψ_x_r[i,j,k] + a_x_hr[i]*(pcur[ii+1,j,k] - pcur[ii,j,k])*_dx
            end
        end
    end
    # y boundaries
    for k = 1:nz
        for j = 1:halo+1
            for i = 1:nx
                jj = j + ny - halo - 2  # shift for bottom boundary pressure indices
                # top boundary
                ψ_y_l[i,j,k] = b_K_y_hl[j] * ψ_y_l[i,j,k] + a_y_hl[j]*(pcur[i, j+1,k] - pcur[i, j,k])*_dy
                # bottom boundary
                ψ_y_r[i,j,k] = b_K_y_hr[j] * ψ_y_r[i,j,k] + a_y_hr[j]*(pcur[i,jj+1,k] - pcur[i,jj,k])*_dy
            end
        end
    end
    # z boundaries
    for k = 1:halo+1
        for j = 1:ny
            for i = 1:nx
                kk = k + nz - halo - 2  # shift for back boundary pressure indices
                # front boundary
                ψ_z_l[i,j,k] = b_K_z_hl[k] * ψ_z_l[i,j,k] + a_z_hl[k]*(pcur[i,j, k+1] - pcur[i, j,k])*_dz
                # back boundary
                ψ_z_r[i,j,k] = b_K_z_hr[k] * ψ_z_r[i,j,k] + a_z_hr[k]*(pcur[i,j,kk+1] - pcur[i,j,kk])*_dz
            end
        end
    end
end

@views function update_p!(pold, pcur, pnew, halo, fact, _dx, _dx2, _dy, _dy2, _dz, _dz2,
                          ψ_x_l = nothing, ψ_x_r = nothing, ψ_y_l = nothing, ψ_y_r = nothing, ψ_z_l = nothing, ψ_z_r = nothing,
                          ξ_x_l = nothing, ξ_x_r = nothing, ξ_y_l = nothing, ξ_y_r = nothing, ξ_z_l = nothing, ξ_z_r = nothing,
                          a_x_l = nothing, a_x_r = nothing, b_K_x_l = nothing, b_K_x_r = nothing,
                          a_y_l = nothing, a_y_r = nothing, b_K_y_l = nothing, b_K_y_r = nothing,
                          a_z_l = nothing, a_z_r = nothing, b_K_z_l = nothing, b_K_z_r = nothing)
    nx, ny, nz = size(pcur)
    for k = 2:nz-1
        for j = 2:ny-1
            for i = 2:nx-1
                d2p_dx2 = (pcur[i+1,j,k] - 2.0*pcur[i,j,k] + pcur[i-1,j,k])*_dx2
                d2p_dy2 = (pcur[i,j+1,k] - 2.0*pcur[i,j,k] + pcur[i,j-1,k])*_dy2
                d2p_dz2 = (pcur[i,j,k+1] - 2.0*pcur[i,j,k] + pcur[i,j,k-1])*_dz2

                damp = 0.0
                # x boundaries
                if i <= halo+1
                    # left boundary
                    dψ_x_dx = (ψ_x_l[i,j,k] - ψ_x_l[i-1,j,k])*_dx
                    ξ_x_l[i-1,j,k] = b_K_x_l[i-1] * ξ_x_l[i-1,j,k] + a_x_l[i-1] * (d2p_dx2 + dψ_x_dx)
                    damp += fact[i,j,k] * (dψ_x_dx + ξ_x_l[i-1,j,k])
                elseif i >= nx - halo
                    # right boundary
                    ii = i - (nx - halo) + 2
                    dψ_x_dx = (ψ_x_r[ii,j,k] - ψ_x_r[ii-1,j,k])*_dx
                    ξ_x_r[ii-1,j,k] = b_K_x_r[ii-1] * ξ_x_r[ii-1,j,k] + a_x_r[ii-1] * (d2p_dx2 + dψ_x_dx)
                    damp += fact[i,j,k] * (dψ_x_dx + ξ_x_r[ii-1,j,k])
                end
                # y boundaries
                if j <= halo+1
                    # top boundary
                    dψ_y_dy = (ψ_y_l[i,j,k] - ψ_y_l[i,j-1,k])*_dy
                    ξ_y_l[i,j-1,k] = b_K_y_l[j-1] * ξ_y_l[i,j-1,k] + a_y_l[j-1] * (d2p_dy2 + dψ_y_dy)
                    damp += fact[i,j,k] * (dψ_y_dy + ξ_y_l[i,j-1,k])
                elseif j >= ny - halo
                    # bottom boundary
                    jj = j - (ny - halo) + 2
                    dψ_y_dy = (ψ_y_r[i,jj,k] - ψ_y_r[i,jj-1,k])*_dy
                    ξ_y_r[i,jj-1,k] = b_K_y_r[jj-1] * ξ_y_r[i,jj-1,k] + a_y_r[jj-1] * (d2p_dy2 + dψ_y_dy)
                    damp += fact[i,j,k] * (dψ_y_dy + ξ_y_r[i,jj-1,k])
                end
                # z boundaries
                if k <= halo+1
                    # front boundary
                    dψ_z_dz = (ψ_z_l[i,j,k] - ψ_z_l[i,j,k-1])*_dz
                    ξ_z_l[i,j,k-1] = b_K_z_l[k-1] * ξ_z_l[i,j,k-1] + a_z_l[k-1] * (d2p_dz2 + dψ_z_dz)
                    damp += fact[i,j,k] * (dψ_z_dz + ξ_z_l[i,j,k-1])
                elseif k >= nz - halo
                    # left boundary
                    kk = k - (nz - halo) + 2
                    dψ_z_dz = (ψ_z_r[i,j,kk] - ψ_z_r[i,j,kk-1])*_dz
                    ξ_z_r[i,j,kk-1] = b_K_z_r[kk-1] * ξ_z_r[i,j,kk-1] + a_z_r[kk-1] * (d2p_dz2 + dψ_z_dz)
                    damp += fact[i,j,k] * (dψ_z_dz + ξ_z_r[i,j,kk-1])
                end

                # update pressure
                pnew[i,j,k] = 2.0 * pcur[i,j,k] - pold[i,j,k] + fact[i,j,k] * (d2p_dx2 + d2p_dy2 + d2p_dz2) + damp
            end
        end
    end
end

@views function inject_sources!(pnew, dt2srctf, possrcs, it)
    _, nsrcs = size(dt2srctf)
    for s = 1:nsrcs
        isrc = possrcs[s,1]
        jsrc = possrcs[s,2]
        ksrc = possrcs[s,2]
        pnew[isrc,jsrc,ksrc] += dt2srctf[it,s]
    end
end

@views function kernel!(
    pold, pcur, pnew, fact, _dx, _dx2, _dy, _dy2, _dz, _dz2,
    halo, ψ_x_l, ψ_x_r, ξ_x_l, ξ_x_r, ψ_y_l, ψ_y_r, ξ_y_l, ξ_y_r, ψ_z_l, ψ_z_r, ξ_z_l, ξ_z_r,
    a_x_hl, a_x_hr, b_K_x_hl, b_K_x_hr,
    a_x_l, a_x_r, b_K_x_l, b_K_x_r,
    a_y_hl, a_y_hr, b_K_y_hl, b_K_y_hr,
    a_y_l, a_y_r, b_K_y_l, b_K_y_r,
    a_z_hl, a_z_hr, b_K_z_hl, b_K_z_hr,
    a_z_l, a_z_r, b_K_z_l, b_K_z_r,
    possrcs, dt2srctf, it
)
    update_ψ!(ψ_x_l, ψ_x_r, ψ_y_l, ψ_y_r, ψ_z_l, ψ_z_r, pcur,
              halo, _dx, _dy, _dz,
              a_x_hl, a_x_hr,
              b_K_x_hl, b_K_x_hr,
              a_y_hl, a_y_hr,
              b_K_y_hl, b_K_y_hr,
              a_z_hl, a_z_hr,
              b_K_z_hl, b_K_z_hr)
    update_p!(pold, pcur, pnew, halo, fact, _dx, _dx2, _dy, _dy2, _dz, _dz2,
              ψ_x_l, ψ_x_r, ψ_y_l, ψ_y_r, ψ_z_l, ψ_z_r,
              ξ_x_l, ξ_x_r, ξ_y_l, ξ_y_r, ξ_z_l, ξ_z_r,
              a_x_l, a_x_r, b_K_x_l, b_K_x_r,
              a_y_l, a_y_r, b_K_y_l, b_K_y_r,
              a_z_l, a_z_r, b_K_z_l, b_K_z_r)
    inject_sources!(pnew, dt2srctf, possrcs, it)

    return pcur, pnew, pold
end

@views function acoustic3D(
    lx::Real,
    ly::Real,
    lz::Real,
    nt::Integer,
    vel::Array{<:Real, 3},
    possrcs;
    halo::Integer = 20,
    rcoef::Real = 0.0001,
    do_vis::Bool = true,
    do_save::Bool = false,
    do_bench::Bool = false,
    nvis::Integer = 5,
    nsave::Integer = 100,
    gif_name::String = "acoustic3D",
    plims::Vector{<:Real} = [-3, 3],
    threshold::Real = 0.01,
    freetop::Bool = true
)
    # Physics
    f0 = 8.0                            # dominating frequency [Hz]
    t0 = 1.2 / f0                       # activation time [s]
    # Derived physics
    vel_max = maximum(vel)              # maximum velocity [m/s]
    # Numerics
    nx, ny, nz    = size(vel)         # number of grid points
    npower        = 2.0
    K_max         = 1.0
    # Derived numerics
    dx = lx / (nx-1)                    # grid step size [m]
    dy = ly / (ny-1)                    # grid step size [m]
    dz = lz / (nz-1)                    # grid step size [m]
    dt = 0.0012                         # 1.0 / (sqrt(1.0/(dx^2) + 1.0/(dy^2) + 1.0/(dz^2))) / vel_max  # timestep size (CFL + Courant condition) [s]
    times = collect(range(0.0,step=dt,length=nt))   # time vector [s]
    # CPML numerics
    alpha_max        = 2.0*π*(f0/2.0)
    thickness_cpml_x = halo * dx
    thickness_cpml_y = halo * dy
    thickness_cpml_z = halo * dz
    d0_x             = -(npower + 1) * vel_max * log(rcoef) / (2.0 * thickness_cpml_x)
    d0_y             = -(npower + 1) * vel_max * log(rcoef) / (2.0 * thickness_cpml_y)
    d0_z             = -(npower + 1) * vel_max * log(rcoef) / (2.0 * thickness_cpml_z)
    a_x_l , a_x_r , b_K_x_l , b_K_x_r  = calc_Kab_CPML(halo,dt,npower,d0_x,alpha_max,K_max,"ongrd")
    a_x_hl, a_x_hr, b_K_x_hl, b_K_x_hr = calc_Kab_CPML(halo,dt,npower,d0_x,alpha_max,K_max,"halfgrd")
    a_y_l , a_y_r , b_K_y_l , b_K_y_r  = calc_Kab_CPML(halo,dt,npower,d0_y,alpha_max,K_max,"ongrd")
    a_y_hl, a_y_hr, b_K_y_hl, b_K_y_hr = calc_Kab_CPML(halo,dt,npower,d0_y,alpha_max,K_max,"halfgrd")
    a_z_l , a_z_r , b_K_z_l , b_K_z_r  = calc_Kab_CPML(halo,dt,npower,d0_z,alpha_max,K_max,"ongrd")
    a_z_hl, a_z_hr, b_K_z_hl, b_K_z_hr = calc_Kab_CPML(halo,dt,npower,d0_z,alpha_max,K_max,"halfgrd")
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
    _dz = 1.0 / dz
    _dz2 = 1.0 / (dz^2)
    fact = (dt^2) .* vel.^2

    # assertions for stability
    @assert max(dx, dy, dz) <= vel_max/(10.0 * f0)   # at least 10pts per wavelength
    
    # Array initialization

    # pressure arrays
    pold = zeros(nx,ny,nz)
    pcur = zeros(nx,ny,nz)
    pnew = zeros(nx,ny,nz)
    # CPML arrays
    ψ_x_l, ψ_x_r = zeros(halo+1,ny,nz), zeros(halo+1,ny,nz)
    ξ_x_l, ξ_x_r = zeros(halo,ny,nz), zeros(halo,ny,nz)
    ψ_y_l, ψ_y_r = zeros(nx,halo+1,nz), zeros(nx,halo+1,nz)
    ξ_y_l, ξ_y_r = zeros(nx,halo,nz), zeros(nx,halo,nz)
    ψ_z_l, ψ_z_r = zeros(nx,ny,halo+1), zeros(nx,ny,halo+1)
    ξ_z_l, ξ_z_r = zeros(nx,ny,halo), zeros(nx,ny,halo)
    # source time functions
    nsrcs = size(possrcs,1)
    dt2srctf = zeros(nt,nsrcs)
    for s = 1:nsrcs
        dt2srctf[:,s] .= (dt^2) .* 1000 .* rickersource1D(times, t0, f0)
    end

    # benchmarking instead of actual computation
    if do_bench
        # run benchmark trial
        trial = @benchmark $kernel!(
            $pold,    $pcur,    $pnew,      $fact, $_dx, $_dx2, $_dy, $_dy2, $_dz, $_dz2,
            $halo,    $ψ_x_l,   $ψ_x_r,     $ξ_x_l, $ξ_x_r, $ψ_y_l, $ψ_y_r, $ξ_y_l, $ξ_y_r, $ψ_z_l, $ψ_z_r, $ξ_z_l, $ξ_z_r,
            $a_x_hl,  $a_x_hr,  $b_K_x_hl,  $b_K_x_hr,
            $a_x_l,   $a_x_r,   $b_K_x_l,   $b_K_x_r,
            $a_y_hl,  $a_y_hr,  $b_K_y_hl,  $b_K_y_hr,
            $a_y_l,   $a_y_r,   $b_K_y_l,   $b_K_y_r,
            $a_z_hl,  $a_z_hr,  $b_K_z_hl,  $b_K_z_hr,
            $a_z_l,   $a_z_r,   $b_K_z_l,   $b_K_z_r,
            $possrcs, $dt2srctf, 1
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
                     3*(nx*ny*nz) +
                     2*((halo+1)*ny*nz) + 2*(halo*ny*nz) +
                     2*((halo+1)*nx*nz) + 2*(halo*nx*nz) +
                     2*((halo+1)*nx*ny) + 2*(halo*nx*ny) +
                     4*3*(halo+1) + 4*3*halo
                    ) * sizeof(Float64) / 1e9
        # effective memory access [GB]
        A_eff = (
            (halo+1)*ny*nz*2*(2 + 1) +         # update_ψ_x!
            (halo+1)*nx*nz*2*(2 + 1) +         # update_ψ_y!
            (halo+1)*nx*ny*2*(2 + 1) +         # update_ψ_z!
            (halo+1)*ny*nz*2*(1 + 1) +         # update ξ_x in update_p!
            (halo+1)*nx*nz*2*(1 + 1) +         # update ξ_y in update_p!
            (halo+1)*nx*ny*2*(1 + 1) +         # update ξ_z in update_p!
            4*nx*ny*nz                         # update_p! (inner points)
        ) * sizeof(Float64) / 1e9
        # effective memory throughput [GB/s]
        T_eff = A_eff / t_it
        @printf("size = %dx%dx%d, time = %1.3e sec, Teff = %1.3f GB/s, memory = %1.3f GB\n", nx, ny, nz, t_it, T_eff, alloc_mem)
        return nothing
    end

    # create results folders
    if do_vis || do_save
        mkpath(DOCS_FLD)
        if do_save
            mkpath(TMP_FLD)
        end
    end

    # time loop
    anim = Animation()
    for it=1:nt
        pold, pcur, pnew = kernel!(
            pold, pcur, pnew, fact, _dx, _dx2, _dy, _dy2, _dz, _dz2,
            halo, ψ_x_l, ψ_x_r, ξ_x_l, ξ_x_r, ψ_y_l, ψ_y_r, ξ_y_l, ξ_y_r, ψ_z_l, ψ_z_r, ξ_z_l, ξ_z_r,
            a_x_hl, a_x_hr, b_K_x_hl, b_K_x_hr,
            a_x_l, a_x_r, b_K_x_l, b_K_x_r,
            a_y_hl, a_y_hr, b_K_y_hl, b_K_y_hr,
            a_y_l, a_y_r, b_K_y_l, b_K_y_r,
            a_z_hl, a_z_hr, b_K_z_hl, b_K_z_hr,
            a_z_l, a_z_r, b_K_z_l, b_K_z_r,
            possrcs, dt2srctf, it
        )

        # visualization
        if do_vis && (it % nvis == 0)
            # take index for slice in middle
            slice_index = div(nz, 2, RoundUp)
            # get velocity slice and pressure slice
            vel_slice = copy(vel[:,:,slice_index])
            p_slice = copy(Array(pcur[:,:,slice_index]))
            # velocity model heatmap
            velview = (((vel_slice .- minimum(vel_slice)) ./ (maximum(vel_slice) - minimum(vel_slice)))) .* (plims[2] - plims[1]) .+ plims[1]
            heatmap(0:dx:lx, 0:dy:ly, velview'; c=:grayC, aspect_ratio=:equal)
            # pressure heatmap
            pview = Array(p_slice) .* 1e3
            maxabsp = @sprintf "%e" maximum(abs.(pview))
            @show maxabsp
            pview[(pview .> plims[1] * threshold) .& (pview .< plims[2] * threshold)] .= NaN
            heatmap!(0:dx:lx, 0:dy:ly, pview';
                  xlims=(0,lx),ylims=(0,ly), clims=(plims[1], plims[2]), aspect_ratio=:equal,
                  xlabel="lx", ylabel="ly", clabel="pressure", c=:diverging_bwr_20_95_c54_n256,
                  title="3D Acoustic CPML (nz/2 slice)\n(halo=$(halo), rcoef=$(rcoef), threshold=$(round(threshold * 100, digits=2))%)\n max abs pressure = $(maxabsp)"
            )
            # sources positions
            # filter out sources not on the slice
            filtered_possrcs = possrcs[(possrcs[:,3] .== slice_index),:]
            scatter!((filtered_possrcs[:,1].-1) .* dx, (filtered_possrcs[:,2].-1) .* dy; markershape=:star, markersize=5, color=:red, label="sources in slice")
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

        # save current pressure as HD5 file
        if do_save && (it % nsave == 0)
            # save model sizes
            h5write(joinpath(TMP_FLD, "$(save_name)_it$(it).h5"), "lx", lx)
            h5write(joinpath(TMP_FLD, "$(save_name)_it$(it).h5"), "ly", ly)
            h5write(joinpath(TMP_FLD, "$(save_name)_it$(it).h5"), "lz", lz)
            # save CPML halo size
            h5write(joinpath(TMP_FLD, "$(save_name)_it$(it).h5"), "halo", halo)
            # save pressure
            h5write(joinpath(TMP_FLD, "$(save_name)_it$(it).h5"), "pcur", Array(pcur))
        end
    end
    # save visualization
    if do_vis
        gif(anim, joinpath(DOCS_FLD, "$(gif_name).gif"))
    end

    return pcur
end
