using Plots, Plots.Measures
using BenchmarkTools
using Printf
using HDF5

default(size=(1000, 600), framestyle=:box,grid=false,margin=20pt)

include("../utils.jl")

# folders for results
DOCS_FLD = joinpath(dirname(dirname(@__DIR__)), "docs")
TMP_FLD = joinpath(DOCS_FLD, "tmp")

# Disable interactive visualization
ENV["GKSwstype"]="nul"

using ImplicitGlobalGrid
import MPI

using ParallelStencil
using ParallelStencil.FiniteDifferences3D
const USE_GPU = true
@static if USE_GPU
    using CUDA
    @init_parallel_stencil(CUDA, Float64, 3)
    # Select first device
    device!(collect(devices())[1])
else
    @init_parallel_stencil(Threads, Float64, 3)
end

# global maximum
max_g(A) = (max_l = maximum(A); MPI.Allreduce(max_l, MPI.MAX, MPI.COMM_WORLD))

@parallel_indices (i,j,k) function update_ψ_x_l!(ψ_x_l, pcur,
                                                 halo, _dx, nx,
                                                 a_x_hl, b_K_x_hl)
    ψ_x_l[i,j,k] = b_K_x_hl[i] * ψ_x_l[i,j,k] + a_x_hl[i]*(pcur[ i+1,j,k] - pcur[ i,j,k])*_dx
    return nothing
end

@parallel_indices (i,j,k) function update_ψ_x_r!(ψ_x_r, pcur,
                                                 halo, _dx, nx,
                                                 a_x_hr, b_K_x_hr)
    ii = i + nx - halo - 2  # shift for right boundary pressure indices
    # right boundary
    ψ_x_r[i,j,k] = b_K_x_hr[i] * ψ_x_r[i,j,k] + a_x_hr[i]*(pcur[ii+1,j,k] - pcur[ii,j,k])*_dx

    return nothing
end

@parallel_indices (i,j,k) function update_ψ_y_l!(ψ_y_l, pcur,
                                                 halo, _dy, ny,
                                                 a_y_hl, b_K_y_hl)
    # top boundary
    ψ_y_l[i,j,k] = b_K_y_hl[j] * ψ_y_l[i,j,k] + a_y_hl[j]*(pcur[i, j+1,k] - pcur[i, j,k])*_dy

    return nothing
end

@parallel_indices (i,j,k) function update_ψ_y_r!(ψ_y_r, pcur,
                                                 halo, _dy, ny,
                                                 a_y_hr, b_K_y_hr)
    jj = j + ny - halo - 2  # shift for bottom boundary pressure indices
    # bottom boundary
    ψ_y_r[i,j,k] = b_K_y_hr[j] * ψ_y_r[i,j,k] + a_y_hr[j]*(pcur[i,jj+1,k] - pcur[i,jj,k])*_dy

    return nothing
end

@parallel_indices (i,j,k) function update_ψ_z_l!(ψ_z_l, pcur,
                                                 halo, _dz, nz,
                                                 a_z_hl, b_K_z_hl)
    # front boundary
    ψ_z_l[i,j,k] = b_K_z_hl[k] * ψ_z_l[i,j,k] + a_z_hl[k]*(pcur[i,j, k+1] - pcur[i,j, k])*_dz

    return nothing
end

@parallel_indices (i,j,k) function update_ψ_z_r!(ψ_z_r, pcur,
                                                 halo, _dz, nz,
                                                 a_z_hr, b_K_z_hr)
    kk = k + nz - halo - 2  # shift for back boundary pressure indices
    # back boundary
    ψ_z_r[i,j,k] = b_K_z_hr[k] * ψ_z_r[i,j,k] + a_z_hr[k]*(pcur[i,j,kk+1] - pcur[i,j,kk])*_dz

    return nothing
end

@parallel_indices (i,j,k) function update_p!(pold, pcur, pnew, halo, fact,
                                             _dx, _dx2, _dy, _dy2, _dz, _dz2, nx, ny, nz,
                                             ψ_x_l, ψ_x_r, ψ_y_l, ψ_y_r, ψ_z_l, ψ_z_r,
                                             ξ_x_l, ξ_x_r, ξ_y_l, ξ_y_r, ξ_z_l, ξ_z_r,
                                             a_x_l, a_x_r, b_K_x_l, b_K_x_r,
                                             a_y_l, a_y_r, b_K_y_l, b_K_y_r,
                                             a_z_l, a_z_r, b_K_z_l, b_K_z_r,
                                             ishift, jshift, kshift,
                                             gnx, gny, gnz)
    # check inside domain
    if i >= 2 && i <= nx-1 && j >= 2 && j <= ny-1 && k >= 2 && k <= nz-1
        # pressure derivatives in space
        d2p_dx2 = (pcur[i+1,j,k] - 2.0*pcur[i,j,k] + pcur[i-1,j,k])*_dx2
        d2p_dy2 = (pcur[i,j+1,k] - 2.0*pcur[i,j,k] + pcur[i,j-1,k])*_dy2
        d2p_dz2 = (pcur[i,j,k+1] - 2.0*pcur[i,j,k] + pcur[i,j,k-1])*_dz2

        damp = 0.0
        # x boundaries
        if i + ishift <= halo+1
            # left boundary
            dψ_x_dx = (ψ_x_l[i,j,k] - ψ_x_l[i-1,j,k])*_dx
            ξ_x_l[i-1,j,k] = b_K_x_l[i-1] * ξ_x_l[i-1,j,k] + a_x_l[i-1] * (d2p_dx2 + dψ_x_dx)
            damp += fact[i,j,k] * (dψ_x_dx + ξ_x_l[i-1,j,k])
        elseif i + ishift >= gnx - halo
            # right boundary
            ii = i - (nx - halo) + 2
            dψ_x_dx = (ψ_x_r[ii,j,k] - ψ_x_r[ii-1,j,k])*_dx
            ξ_x_r[ii-1,j,k] = b_K_x_r[ii-1] * ξ_x_r[ii-1,j,k] + a_x_r[ii-1] * (d2p_dx2 + dψ_x_dx)
            damp += fact[i,j,k] * (dψ_x_dx + ξ_x_r[ii-1,j,k])
        end
        # y boundaries
        if j + jshift <= halo+1
            # left boundary
            dψ_y_dy = (ψ_y_l[i,j,k] - ψ_y_l[i,j-1,k])*_dy
            ξ_y_l[i,j-1,k] = b_K_y_l[j-1] * ξ_y_l[i,j-1,k] + a_y_l[j-1] * (d2p_dy2 + dψ_y_dy)
            damp += fact[i,j,k] * (dψ_y_dy + ξ_y_l[i,j-1,k])
        elseif j + jshift >= gny - halo
            # right boundary
            jj = j - (ny - halo) + 2
            dψ_y_dy = (ψ_y_r[i,jj,k] - ψ_y_r[i,jj-1,k])*_dy
            ξ_y_r[i,jj-1,k] = b_K_y_r[jj-1] * ξ_y_r[i,jj-1,k] + a_y_r[jj-1] * (d2p_dy2 + dψ_y_dy)
            damp += fact[i,j,k] * (dψ_y_dy + ξ_y_r[i,jj-1,k])
        end
        # z boundaries
        if k + kshift <= halo+1
            # left boundary
            dψ_z_dz = (ψ_z_l[i,j,k] - ψ_z_l[i,j,k-1])*_dz
            ξ_z_l[i,j,k-1] = b_K_z_l[k-1] * ξ_z_l[i,j,k-1] + a_z_l[k-1] * (d2p_dz2 + dψ_z_dz)
            damp += fact[i,j,k] * (dψ_z_dz + ξ_z_l[i,j,k-1])
        elseif k + kshift >= gnz - halo
            # right boundary
            kk = k - (nz - halo) + 2
            dψ_z_dz = (ψ_z_r[i,j,kk] - ψ_z_r[i,j,kk-1])*_dz
            ξ_z_r[i,j,kk-1] = b_K_z_r[kk-1] * ξ_z_r[i,j,kk-1] + a_z_r[kk-1] * (d2p_dz2 + dψ_z_dz)
            damp += fact[i,j,k] * (dψ_z_dz + ξ_z_r[i,j,kk-1])
        end

        # update pressure
        pnew[i,j,k] = 2.0 * pcur[i,j,k] - pold[i,j,k] + fact[i,j,k] * (d2p_dx2 + d2p_dy2 + d2p_dz2) + damp
    end

    return nothing
end

@parallel_indices (is) function inject_sources!(pnew, dt2srctf, possrcs, it, ishift, jshift, kshift, nx, ny, nz)
    # Get local source positions from global ones
    isrc = floor(Int, possrcs[is,1]) - ishift
    jsrc = floor(Int, possrcs[is,2]) - jshift
    ksrc = floor(Int, possrcs[is,3]) - kshift

    # Check if source is inside local grid
    if (1 <= isrc <= nx) && (1 <= jsrc <= ny) && (1 <= ksrc <= nz)
        pnew[isrc,jsrc,ksrc] += dt2srctf[it,is]
    end

    return nothing
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
    possrcs, dt2srctf, it,
    gnx, gny, gnz,
    dims, coords, b_width
)
    nx, ny, nz = size(pcur)
    # compute shifting to
    ishift = coords[1] * (nx-2)
    jshift = coords[2] * (ny-2)
    kshift = coords[3] * (nz-2)

    # update ψ arrays
    if coords[1] == 0
        @parallel_async (1:halo+1,1:ny,1:nz) update_ψ_x_l!(ψ_x_l, pcur,
                                                           halo, _dx, nx,
                                                           a_x_hl, b_K_x_hl)
    end
    if coords[1] == dims[1]-1
        @parallel_async (1:halo+1,1:ny,1:nz) update_ψ_x_r!(ψ_x_r, pcur,
                                                           halo, _dx, nx,
                                                           a_x_hr, b_K_x_hr)
    end
    if coords[2] == 0
        @parallel_async (1:nx,1:halo+1,1:nz) update_ψ_y_l!(ψ_y_l, pcur,
                                                           halo, _dy, ny,
                                                           a_y_hl, b_K_y_hl)
    end
    if coords[2] == dims[2]-1
        @parallel_async (1:nx,1:halo+1,1:nz) update_ψ_y_r!(ψ_y_r, pcur,
                                                           halo, _dy, ny,
                                                           a_y_hr, b_K_y_hr)
    end
    if coords[3] == 0
        @parallel_async (1:nx,1:ny,1:halo+1) update_ψ_z_l!(ψ_z_l, pcur,
                                                           halo, _dz, nz,
                                                           a_z_hl, b_K_z_hl)
    end
    if coords[3] == dims[3]-1
        @parallel_async (1:nx,1:ny,1:halo+1) update_ψ_z_r!(ψ_z_r, pcur,
                                                           halo, _dz, nz,
                                                           a_z_hr, b_K_z_hr)
    end
    @synchronize

    @hide_communication b_width begin
         # update presure and ξ arrays
        @parallel update_p!(pold, pcur, pnew, halo, fact,
                            _dx, _dx2, _dy, _dy2, _dz, _dz2, nx, ny, nz,
                            ψ_x_l, ψ_x_r, ψ_y_l, ψ_y_r, ψ_z_l, ψ_z_r,
                            ξ_x_l, ξ_x_r, ξ_y_l, ξ_y_r, ξ_z_l, ξ_z_r,
                            a_x_l, a_x_r, b_K_x_l, b_K_x_r,
                            a_y_l, a_y_r, b_K_y_l, b_K_y_r,
                            a_z_l, a_z_r, b_K_z_l, b_K_z_r,
                            ishift, jshift, kshift,
                            gnx, gny, gnz)
        # exchange new pressure
        update_halo!(pnew)
    end

    # inject sources
    @parallel (1:size(possrcs,1)) inject_sources!(pnew, dt2srctf, possrcs, it, ishift, jshift, kshift, nx, ny, nz)

    return pcur, pnew, pold
end

@views function acoustic3D_multixPU(
    lx::Real,
    ly::Real,
    lz::Real,
    nx::Integer,
    ny::Integer,
    nz::Integer,
    nt::Integer,
    vel_func::Function,
    possrcs;
    halo::Integer = 20,
    rcoef::Real = 0.0001,
    do_vis::Bool = true,
    do_save::Bool = true,
    nvis::Integer = 5,
    nsave::Integer = 100,
    gif_name::String = "acoustic3D_multixPU",
    save_name::String = "acoustic3D_multixPU",
    plims::Vector{<:Real} = [-3, 3],
    threshold::Real = 0.01,
    freetop::Bool = true,
    init_MPI::Bool = true
)
    # Initialize global grid
    me, dims, nprocs, coords, comm_cart = init_global_grid(nx, ny, nz; init_MPI=init_MPI)
    b_width = (8, 8, 4)                 # hide communication parameters
    # Physics
    f0 = 8.0                            # dominating frequency [Hz]
    t0 = 1.2 / f0                       # activation time [s]
    # Numerics
    npower        = 2.0
    K_max         = 1.0
    # Derived numerics
    gnx, gny, gnz = nx_g(), ny_g(), nz_g()  # global grid size
    dx = lx / (gnx-1)                    # grid step size x-direction [m]
    dy = ly / (gny-1)                    # grid step size y-direction [m]
    dz = lz / (gnz-1)                    # grid step size z-direction [m]
    dt = 0.0012                         # 1.0 / (sqrt(1.0/(dx^2) + 1.0/(dy^2) + 1.0/(dz^2))) / vel_max  # timestep size (CFL + Courant condition) [s]
    times = collect(range(0.0,step=dt,length=nt))   # time vector [s]
    # Initialize local velocity model
    vel  = zeros(nx, ny, nz)
    vel .= [ vel_func(x_g(ix, dx, vel), y_g(iy, dy, vel), z_g(iz, dz, vel)) for ix=1:nx,iy=1:ny,iz=1:nz ]
    # Derived physics
    vel_max = max_g(vel)                # maximum velocity [m/s]
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
    fact = Data.Array( (dt^2) .* vel.^2 )

    # assertions for stability
    if me == 0 
        @assert max(dx, dy, dz) <= vel_max / (10.0 * f0)   # at least 10pts per wavelength
    end
    
    # Array initialization

    # pressure arrays
    pold = @zeros(nx,ny,nz)
    pcur = @zeros(nx,ny,nz)
    pnew = @zeros(nx,ny,nz)
    # CPML arrays
    a_x_l , a_x_r , b_K_x_l , b_K_x_r = Data.Array(a_x_l), Data.Array(a_x_r), Data.Array(b_K_x_l), Data.Array(b_K_x_r)
    a_x_hl , a_x_hr , b_K_x_hl , b_K_x_hr = Data.Array(a_x_hl), Data.Array(a_x_hr), Data.Array(b_K_x_hl), Data.Array(b_K_x_hr)
    a_y_l , a_y_r , b_K_y_l , b_K_y_r = Data.Array(a_y_l), Data.Array(a_y_r), Data.Array(b_K_y_l), Data.Array(b_K_y_r)
    a_y_hl , a_y_hr , b_K_y_hl , b_K_y_hr = Data.Array(a_y_hl), Data.Array(a_y_hr), Data.Array(b_K_y_hl), Data.Array(b_K_y_hr)
    a_z_l , a_z_r , b_K_z_l , b_K_z_r = Data.Array(a_z_l), Data.Array(a_z_r), Data.Array(b_K_z_l), Data.Array(b_K_z_r)
    a_z_hl , a_z_hr , b_K_z_hl , b_K_z_hr = Data.Array(a_z_hl), Data.Array(a_z_hr), Data.Array(b_K_z_hl), Data.Array(b_K_z_hr)
    ψ_x_l, ψ_x_r = @zeros(halo+1,ny,nz), @zeros(halo+1,ny,nz)
    ξ_x_l, ξ_x_r = @zeros(halo,ny,nz), @zeros(halo,ny,nz)
    ψ_y_l, ψ_y_r = @zeros(nx,halo+1,nz), @zeros(nx,halo+1,nz)
    ξ_y_l, ξ_y_r = @zeros(nx,halo,nz), @zeros(nx,halo,nz)
    ψ_z_l, ψ_z_r = @zeros(nx,ny,halo+1), @zeros(nx,ny,halo+1)
    ξ_z_l, ξ_z_r = @zeros(nx,ny,halo), @zeros(nx,ny,halo)
    # source time functions
    nsrcs = size(possrcs,1)
    dt2srctf = zeros(nt,nsrcs)
    for s = 1:nsrcs
        dt2srctf[:,s] .= (dt^2) .* 1000 .* rickersource1D(times, t0, f0)
    end
    possrcs_a = Data.Array( possrcs )
    dt2srctf = Data.Array( dt2srctf )

    # create results folders
    if me == 0 && (do_vis || do_save)
        mkpath(DOCS_FLD)
        if do_save
            rm(TMP_FLD, recursive=true, force=true)
            mkpath(TMP_FLD)
        end
    end

    # create animation and allocate arrays for global array
    if do_vis
        if me == 0 anim = Animation() end
        nx_v,ny_v,nz_v = (nx-2)*dims[1],(ny-2)*dims[2],(nz-2)*dims[3]
        if (nx_v*ny_v*nz_v*sizeof(Data.Number) > 0.8*Sys.free_memory()) error("Not enough memory for saving.") end
        pcur_global = zeros(nx_v, ny_v, nz_v) # global array for saving
        vel_global = zeros(nx_v, ny_v, nz_v)
        pcur_inner = zeros(nx-2, ny-2, nz-2) # no halo local array for saving
        vel_inner = zeros(nx-2, ny-2, nz-2) # no halo local array for saving
        vel_inner .= vel[2:end-1,2:end-1,2:end-1]
        gather!(vel_inner, vel_global)
    end

    # disable garbage collection
    GC.gc(); GC.enable(false)
    # time for benchmark
    t_tic = 0.0; niter = 0
    # time loop
    for it=1:nt
        # skip first 20 iterations
        if (it==200) t_tic = tic(); niter = 0 end
        pold, pcur, pnew = kernel!(
            pold, pcur, pnew, fact, _dx, _dx2, _dy, _dy2, _dz, _dz2,
            halo, ψ_x_l, ψ_x_r, ξ_x_l, ξ_x_r, ψ_y_l, ψ_y_r, ξ_y_l, ξ_y_r, ψ_z_l, ψ_z_r, ξ_z_l, ξ_z_r,
            a_x_hl, a_x_hr, b_K_x_hl, b_K_x_hr,
            a_x_l, a_x_r, b_K_x_l, b_K_x_r,
            a_y_hl, a_y_hr, b_K_y_hl, b_K_y_hr,
            a_y_l, a_y_r, b_K_y_l, b_K_y_r,
            a_z_hl, a_z_hr, b_K_z_hl, b_K_z_hr,
            a_z_l, a_z_r, b_K_z_l, b_K_z_r,
            possrcs_a, dt2srctf, it,
            gnx, gny, gnz,
            dims, coords, b_width
        )

        niter += 1

        # visualization
        if do_vis && (it % nvis == 0)
            pcur_inner .= Array(pcur)[2:end-1,2:end-1,2:end-1]; gather!(pcur_inner, pcur_global)
        end
        if me == 0 && do_vis && (it % nvis == 0)
            # take index for slice in middle
            slice_index = div(nz_g()-2, 2, RoundUp)
            # get velocity slice and pressure slice
            vel_slice = vel_global[:,:,slice_index]
            p_slice = pcur_global[:,:,slice_index]
            # velocity model heatmap
            velview = (((vel_slice .- minimum(vel_slice)) ./ (maximum(vel_slice) - minimum(vel_slice)))) .* (plims[2] - plims[1]) .+ plims[1]
            heatmap(dx:dx:lx-dx, dy:dy:ly-dy, velview'; c=:grayC, aspect_ratio=:equal)
            # pressure heatmap
            pview = p_slice .* 1e3
            maxabsp = @sprintf "%e" maximum(abs.(pview))
            @show maxabsp
            pview[(pview .> plims[1] * threshold) .& (pview .< plims[2] * threshold)] .= NaN
            heatmap!(dx:dx:lx-dx, dy:dy:ly-dy, pview';
                  xlims=(0,lx), ylims=(0,ly), clims=(plims[1], plims[2]), aspect_ratio=:equal,
                  xlabel="lx", ylabel="ly", clabel="pressure", c=:diverging_bwr_20_95_c54_n256,
                  title="3D Acoustic CPML (nz/2 slice)\n(halo=$(halo), rcoef=$(rcoef), threshold=$(round(threshold * 100, digits=2))%)\n max abs pressure = $(maxabsp)"
            )
            # sources positions
            # filter out sources not on the slice
            filtered_possrcs = possrcs[(possrcs[:,3] .== slice_index + 1),:]
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
            h5write(joinpath(TMP_FLD, "$(save_name)_it$(it)_proc$(me).h5"), "lx", lx)
            h5write(joinpath(TMP_FLD, "$(save_name)_it$(it)_proc$(me).h5"), "ly", ly)
            h5write(joinpath(TMP_FLD, "$(save_name)_it$(it)_proc$(me).h5"), "lz", lz)
            # save CPML halo size
            h5write(joinpath(TMP_FLD, "$(save_name)_it$(it)_proc$(me).h5"), "halo", halo)
            # save pressure
            h5write(joinpath(TMP_FLD, "$(save_name)_it$(it)_proc$(me).h5"), "pcur", Array(pcur))
        end
    end
    # reenable garbage collection
    GC.enable(true)

    # compute performance
    t_toc = toc()
    t_it  = t_toc / niter                  # Execution time per iteration [s]
    # allocated memory [GB]
    local_alloc_mem = (
        3*(nx*ny*nz) +
        2*((halo+1)*ny*nz) + 2*(halo*ny*nz) +
        2*((halo+1)*nx*nz) + 2*(halo*nx*nz) +
        2*((halo+1)*nx*ny) + 2*(halo*nx*ny) +
        4*3*(halo+1) + 4*3*halo
    ) * sizeof(Float64) / 1e9
    global_alloc_mem = nprocs * local_alloc_mem
    # effective memory access [GB]
    A_eff = (
        (halo+1)*gny*gnz*2*(2 + 1) +         # update_ψ_x!
        (halo+1)*gnx*gnz*2*(2 + 1) +         # update_ψ_y!
        (halo+1)*gnx*gny*2*(2 + 1) +         # update_ψ_z!
        (halo+1)*gny*gnz*2*(1 + 1) +         # update ξ_x in update_p!
        (halo+1)*gnx*gnz*2*(1 + 1) +         # update ξ_y in update_p!
        (halo+1)*gnx*gny*2*(1 + 1) +         # update ξ_z in update_p!
        4*gnx*gny*gnz                        # update_p! (inner points)
    ) * sizeof(Float64) / 1e9
    # effective memory throughput [GB/s]
    T_eff = A_eff / t_it

    if me == 0 @printf("size = %dx%dx%d, total time = %1.3e sec, time per it = %1.3e sec, Teff = %1.3f GB/s, local memory = %1.3f GB, global memory = %1.3f GB\n", nx, ny, nz, t_toc, t_it, T_eff, local_alloc_mem, global_alloc_mem) end
    
    # save visualization
    if me == 0 && do_vis
        gif(anim, joinpath(DOCS_FLD, "$(gif_name).gif"))
    end

    finalize_global_grid(;finalize_MPI=init_MPI)

    return nothing
end
