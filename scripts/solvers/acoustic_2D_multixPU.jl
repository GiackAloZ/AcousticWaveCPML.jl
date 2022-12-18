using Plots, Plots.Measures
using BenchmarkTools
using Printf

default(size=(1000, 600), framestyle=:box,grid=false,margin=20pt)

include("../utils.jl")

# folders for results
DOCS_FLD = joinpath(dirname(dirname(@__DIR__)), "docs")

# Disable interactive visualization
ENV["GKSwstype"]="nul"

using ImplicitGlobalGrid
import MPI

using ParallelStencil
using ParallelStencil.FiniteDifferences2D
const USE_GPU = true
@static if USE_GPU
    using CUDA
    @init_parallel_stencil(CUDA, Float64, 2)
    # Disable interactive visualization on GPU environment
    ENV["GKSwstype"]="nul"
    # Select first device
    device!(collect(devices())[1])
else
    @init_parallel_stencil(Threads, Float64, 2)
end

# global maximum
max_g(A) = (max_l = maximum(A); MPI.Allreduce(max_l, MPI.MAX, MPI.COMM_WORLD))

@parallel_indices (i,j) function update_ψ_x_l!(ψ_x_l, pcur,
                                               halo, _dx, nx,
                                               a_x_hl, b_K_x_hl)
    ψ_x_l[i,j] = b_K_x_hl[i] * ψ_x_l[i,j] + a_x_hl[i]*(pcur[ i+1,j] - pcur[ i,j])*_dx
    return nothing
end

@parallel_indices (i,j) function update_ψ_x_r!(ψ_x_r, pcur,
                                                 halo, _dx, nx,
                                                 a_x_hr, b_K_x_hr)
    ii = i + nx - halo - 2  # shift for right boundary pressure indices
    # right boundary
    ψ_x_r[i,j] = b_K_x_hr[i] * ψ_x_r[i,j] + a_x_hr[i]*(pcur[ii+1,j] - pcur[ii,j])*_dx

    return nothing
end

@parallel_indices (i,j) function update_ψ_y_l!(ψ_y_l, pcur,
                                                 halo, _dy, ny,
                                                 a_y_hl, b_K_y_hl)
    # top boundary
    ψ_y_l[i,j] = b_K_y_hl[j] * ψ_y_l[i,j] + a_y_hl[j]*(pcur[i, j+1] - pcur[i, j])*_dy

    return nothing
end

@parallel_indices (i,j) function update_ψ_y_r!(ψ_y_r, pcur,
                                                 halo, _dy, ny,
                                                 a_y_hr, b_K_y_hr)
    jj = j + ny - halo - 2  # shift for bottom boundary pressure indices
    # bottom boundary
    ψ_y_r[i,j] = b_K_y_hr[j] * ψ_y_r[i,j] + a_y_hr[j]*(pcur[i,jj+1] - pcur[i,jj])*_dy

    return nothing
end

@parallel_indices (i,j) function update_p!(pold, pcur, pnew, halo, fact,
                                             _dx, _dx2, _dy, _dy2, nx, ny,
                                             ψ_x_l, ψ_x_r, ψ_y_l, ψ_y_r,
                                             ξ_x_l, ξ_x_r, ξ_y_l, ξ_y_r,
                                             a_x_l, a_x_r, b_K_x_l, b_K_x_r,
                                             a_y_l, a_y_r, b_K_y_l, b_K_y_r,
                                             ishift, jshift,
                                             gnx, gny)
    # pressure derivatives in space
    d2p_dx2 = (pcur[i+1,j] - 2.0*pcur[i,j] + pcur[i-1,j])*_dx2
    d2p_dy2 = (pcur[i,j+1] - 2.0*pcur[i,j] + pcur[i,j-1])*_dy2

    damp = 0.0
    # x boundaries
    if i + ishift <= halo+1
        # left boundary
        dψ_x_dx = (ψ_x_l[i,j] - ψ_x_l[i-1,j])*_dx
        ξ_x_l[i-1,j] = b_K_x_l[i-1] * ξ_x_l[i-1,j] + a_x_l[i-1] * (d2p_dx2 + dψ_x_dx)
        damp += fact[i,j] * (dψ_x_dx + ξ_x_l[i-1,j])
    elseif i + ishift >= gnx - halo
        # right boundary
        ii = i - (nx - halo) + 2
        dψ_x_dx = (ψ_x_r[ii,j] - ψ_x_r[ii-1,j])*_dx
        ξ_x_r[ii-1,j] = b_K_x_r[ii-1] * ξ_x_r[ii-1,j] + a_x_r[ii-1] * (d2p_dx2 + dψ_x_dx)
        damp += fact[i,j] * (dψ_x_dx + ξ_x_r[ii-1,j])
    end
    # y boundaries
    if j + jshift <= halo+1
        # left boundary
        dψ_y_dy = (ψ_y_l[i,j] - ψ_y_l[i,j-1])*_dy
        ξ_y_l[i,j-1] = b_K_y_l[j-1] * ξ_y_l[i,j-1] + a_y_l[j-1] * (d2p_dy2 + dψ_y_dy)
        damp += fact[i,j] * (dψ_y_dy + ξ_y_l[i,j-1])
    elseif j + jshift >= gny - halo
        # right boundary
        jj = j - (ny - halo) + 2
        dψ_y_dy = (ψ_y_r[i,jj] - ψ_y_r[i,jj-1])*_dy
        ξ_y_r[i,jj-1] = b_K_y_r[jj-1] * ξ_y_r[i,jj-1] + a_y_r[jj-1] * (d2p_dy2 + dψ_y_dy)
        damp += fact[i,j] * (dψ_y_dy + ξ_y_r[i,jj-1])
    end

    # update pressure
    pnew[i,j] = 2.0 * pcur[i,j] - pold[i,j] + fact[i,j] * (d2p_dx2 + d2p_dy2) + damp

    return nothing
end

@parallel_indices (is) function inject_sources!(pnew, dt2srctf, possrcs, it, ishift, jshift, nx, ny)
    # Get local source positions from global ones
    isrc = floor(Int, possrcs[is,1]) - ishift
    jsrc = floor(Int, possrcs[is,2]) - jshift

    # Check if source is inside local grid
    if (1 <= isrc <= nx) && (1 <= jsrc <= ny)
        pnew[isrc,jsrc] += dt2srctf[it,is]
    end

    return nothing
end

@views function kernel!(
    pold, pcur, pnew, fact, _dx, _dx2, _dy, _dy2,
    halo, ψ_x_l, ψ_x_r, ξ_x_l, ξ_x_r, ψ_y_l, ψ_y_r, ξ_y_l, ξ_y_r,
    a_x_hl, a_x_hr, b_K_x_hl, b_K_x_hr,
    a_x_l, a_x_r, b_K_x_l, b_K_x_r,
    a_y_hl, a_y_hr, b_K_y_hl, b_K_y_hr,
    a_y_l, a_y_r, b_K_y_l, b_K_y_r,
    possrcs, dt2srctf, it,
    gnx, gny,
    dims, coords, b_width
)
    nx, ny = size(pcur)
    # compute shifting to
    ishift = coords[1] * (nx-2)
    jshift = coords[2] * (ny-2)

    # update ψ arrays
    if coords[1] == 0
        @parallel_async (1:halo+1,1:ny) update_ψ_x_l!(ψ_x_l, pcur,
                                                           halo, _dx, nx,
                                                           a_x_hl, b_K_x_hl)
    end
    if coords[1] == dims[1]-1
        @parallel_async (1:halo+1,1:ny) update_ψ_x_r!(ψ_x_r, pcur,
                                                           halo, _dx, nx,
                                                           a_x_hr, b_K_x_hr)
    end
    if coords[2] == 0
        @parallel_async (1:nx,1:halo+1) update_ψ_y_l!(ψ_y_l, pcur,
                                                           halo, _dy, ny,
                                                           a_y_hl, b_K_y_hl)
    end
    if coords[2] == dims[2]-1
        @parallel_async (1:nx,1:halo+1) update_ψ_y_r!(ψ_y_r, pcur,
                                                           halo, _dy, ny,
                                                           a_y_hr, b_K_y_hr)
    end
    @synchronize

    @hide_communication b_width begin
         # update presure and ξ arrays
        @parallel (2:nx-1,2:ny-1) update_p!(pold, pcur, pnew, halo, fact,
                                            _dx, _dx2, _dy, _dy2, nx, ny,
                                            ψ_x_l, ψ_x_r, ψ_y_l, ψ_y_r,
                                            ξ_x_l, ξ_x_r, ξ_y_l, ξ_y_r,
                                            a_x_l, a_x_r, b_K_x_l, b_K_x_r,
                                            a_y_l, a_y_r, b_K_y_l, b_K_y_r,
                                            ishift, jshift,
                                            gnx, gny)
        # inject sources
        @parallel (1:size(possrcs,1)) inject_sources!(pnew, dt2srctf, possrcs, it, ishift, jshift, nx, ny)
        # exchange new pressure
        update_halo!(pnew)
    end

    return pcur, pnew, pold
end


@views function acoustic2D_multixPU(
    lx::Real,
    ly::Real,
    nx::Integer,
    ny::Integer,
    nt::Integer,
    vel_func::Function,
    possrcs;
    halo::Integer = 20,
    rcoef::Real = 0.0001,
    do_vis::Bool = true,
    nvis::Integer = 5,
    gif_name::String = "acoustic2D_multixPU",
    plims::Vector{<:Real} = [-3, 3],
    threshold::Real = 0.01,
    freetop::Bool = true,
    init_MPI::Bool = true
)
    # Initialize global grid
    me, dims, nprocs, coords, comm_cart = init_global_grid(nx, ny, 1; init_MPI=init_MPI)
    b_width = (8, 8)                    # hide communication parameters
    # Physics
    f0 = 8.0                            # dominating frequency [Hz]
    t0 = 1.2 / f0                       # activation time [s]
    # Numerics
    npower        = 2.0
    K_max         = 1.0
    # Derived numerics
    gnx, gny = nx_g(), ny_g()            # global grid size
    dx = lx / (gnx-1)                    # grid step size x-direction [m]
    dy = ly / (gny-1)                    # grid step size y-direction [m]
    dt = 0.0012                          # 1.0 / (sqrt(1.0/(dx^2) + 1.0/(dy^2))) / vel_max  # timestep size (CFL + Courant condition) [s]
    times = collect(range(0.0,step=dt,length=nt))   # time vector [s]
    # Initialize local velocity model
    vel  = zeros(nx, ny)
    vel .= [ vel_func(x_g(ix, dx, vel), y_g(iy, dy, vel)) for ix=1:nx,iy=1:ny ]
    # Derived physics
    vel_max = max_g(vel)                # maximum velocity [m/s]
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
    fact = Data.Array( (dt^2) .* vel.^2 )

    # assertions for stability
    if me == 0
        @assert max(dx, dy) <= vel_max / (10.0 * f0)   # at least 10pts per wavelength
    end
    
    # Array initialization

    # pressure arrays
    pold = @zeros(nx,ny)
    pcur = @zeros(nx,ny)
    pnew = @zeros(nx,ny)
    # CPML arrays
    a_x_l , a_x_r , b_K_x_l , b_K_x_r = Data.Array(a_x_l), Data.Array(a_x_r), Data.Array(b_K_x_l), Data.Array(b_K_x_r)
    a_x_hl , a_x_hr , b_K_x_hl , b_K_x_hr = Data.Array(a_x_hl), Data.Array(a_x_hr), Data.Array(b_K_x_hl), Data.Array(b_K_x_hr)
    a_y_l , a_y_r , b_K_y_l , b_K_y_r = Data.Array(a_y_l), Data.Array(a_y_r), Data.Array(b_K_y_l), Data.Array(b_K_y_r)
    a_y_hl , a_y_hr , b_K_y_hl , b_K_y_hr = Data.Array(a_y_hl), Data.Array(a_y_hr), Data.Array(b_K_y_hl), Data.Array(b_K_y_hr)
    ψ_x_l, ψ_x_r = @zeros(halo+1,ny), @zeros(halo+1,ny)
    ξ_x_l, ξ_x_r = @zeros(halo,ny), @zeros(halo,ny)
    ψ_y_l, ψ_y_r = @zeros(nx,halo+1), @zeros(nx,halo+1)
    ξ_y_l, ξ_y_r = @zeros(nx,halo), @zeros(nx,halo)
    # source time functions
    nsrcs = size(possrcs,1)
    dt2srctf = zeros(nt,nsrcs)
    for s = 1:nsrcs
        dt2srctf[:,s] .= (dt^2) .* 1000 .* rickersource1D(times, t0, f0)
    end
    possrcs_a = Data.Array( possrcs )
    dt2srctf = Data.Array( dt2srctf )

    # create results folders
    if do_vis
        mkpath(DOCS_FLD)
    end

    # create animation and allocate arrays for global array
    if do_vis
        if me == 0 anim = Animation() end
        nx_v,ny_v = (nx-2)*dims[1],(ny-2)*dims[2]
        if (nx_v*ny_v*sizeof(Data.Number) > 0.8*Sys.free_memory()) error("Not enough memory for saving.") end
        pcur_global = zeros(nx_v, ny_v) # global array for saving
        vel_global = zeros(nx_v, ny_v)
        pcur_inner = zeros(nx-2, ny-2) # no halo local array for saving
        vel_inner = zeros(nx-2, ny-2) # no halo local array for saving
        vel_inner .= vel[2:end-1,2:end-1]
        gather!(vel_inner, vel_global)
    end

    # disable garbage collection
    GC.gc(); GC.enable(false)
    # time for benchmark
    t_tic = 0.0; niter = 0
    # time loop
    for it=1:nt
        # skip first 200 iterations
        if (it==201) t_tic = tic(); niter = 0 end

        pold, pcur, pnew = kernel!(
            pold, pcur, pnew, fact, _dx, _dx2, _dy, _dy2,
            halo, ψ_x_l, ψ_x_r, ξ_x_l, ξ_x_r, ψ_y_l, ψ_y_r, ξ_y_l, ξ_y_r,
            a_x_hl, a_x_hr, b_K_x_hl, b_K_x_hr,
            a_x_l, a_x_r, b_K_x_l, b_K_x_r,
            a_y_hl, a_y_hr, b_K_y_hl, b_K_y_hr,
            a_y_l, a_y_r, b_K_y_l, b_K_y_r,
            possrcs_a, dt2srctf, it,
            gnx, gny,
            dims, coords, b_width
        )

        niter += 1

        # visualization
        if do_vis && (it % nvis == 0)
            pcur_inner .= Array(pcur)[2:end-1,2:end-1]; gather!(pcur_inner, pcur_global)
        end
        if me == 0 && do_vis && (it % nvis == 0)
            # velocity model heatmap
            velview = (((copy(vel_global) .- minimum(vel_global)) ./ (maximum(vel_global) - minimum(vel_global)))) .* (plims[2] - plims[1]) .+ plims[1]
            heatmap(dx:dx:lx-dx, dy:dy:ly-dy, velview'; c=:grayC, aspect_ratio=:equal)
            # pressure heatmap
            pview = pcur_global .* 1e3
            maxabsp = @sprintf "%e" maximum(abs.(pview))
            @show maxabsp
            pview[(pview .> plims[1] * threshold) .& (pview .< plims[2] * threshold)] .= NaN
            heatmap!(dx:dx:lx-dx, dy:dy:ly-dy, pview';
                  xlims=(0,lx),ylims=(0,ly), clims=(plims[1], plims[2]), aspect_ratio=:equal,
                  xlabel="lx", ylabel="ly", clabel="pressure", c=:diverging_bwr_20_95_c54_n256,
                  title="2D Acoustic CPML\n(halo=$(halo), rcoef=$(rcoef), threshold=$(round(threshold * 100, digits=2))%)\n max abs pressure = $(maxabsp)"
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
    # reenable garbage collection
    GC.enable(true)

    # compute performance
    t_toc = toc()
    t_it  = t_toc / niter                  # Execution time per iteration [s]
    # allocated memory [GB]
    local_alloc_mem = (
        3*(nx*ny) +
        2*((halo+1)*ny) + 2*(halo*ny) +
        2*((halo+1)*nx) + 2*(halo*nx) +
        4*2*(halo+1) + 4*2*halo
    ) * sizeof(Float64) / 1e9
    global_alloc_mem = nprocs * local_alloc_mem
    # effective memory access [GB]
    A_eff = (
    (halo+1)*gny*2*(2 + 1) +         # update_ψ_x!
    (halo+1)*gnx*2*(2 + 1) +         # update_ψ_y!
    (halo+1)*gny*2*(1 + 1) +         # update ξ_x in update_p!
    (halo+1)*gnx*2*(1 + 1) +         # update ξ_y in update_p!
    4*nx*ny                          # update_p! (inner points)
    ) * sizeof(Float64) / 1e9
    # effective memory throughput [GB/s]
    T_eff = A_eff / t_it

    if me == 0 @printf("size = %dx%d, total time = %1.3e sec, time per it = %1.3e sec, Teff = %1.3f GB/s, local memory = %1.3f GB, global memory = %1.3f GB\n", nx, ny, t_toc, t_it, T_eff, local_alloc_mem, global_alloc_mem) end

    # save visualization
    if me == 0 && do_vis
        gif(anim, joinpath(DOCS_FLD, "$(gif_name).gif"))
    end

    finalize_global_grid(;finalize_MPI=init_MPI)

    return nothing
end
