using AcousticWaveCPML
using AcousticWaveCPML.Acoustic3D_Threads

using Logging
errorlogger = ConsoleLogger(stderr, Logging.Error)
global_logger(errorlogger)

include(joinpath(@__DIR__, "models/rescale_model.jl"))

function run_center()
    # simple constant velocity model
    nx = ny = nz = 201                           # grid size
    lx = ly = lz = 2000.0                        # model sizes [m]
    vel = 2000.0 .* ones(Float64, nx, ny, nz)    # velocity model [m/s]
    lt = 2.0                                     # final time [s]
    # sources
    f0 = 10.0                                    # source dominating frequency [Hz]
    t0 = 4 / f0                                  # source activation time [s]
    stf = rickersource1D                         # second derivative of gaussian
    possrcs = zeros(1,3)
    possrcs[1,:] .= [lx/2, ly/2, lz/2]
    srcs = Sources(possrcs, [t0], [stf], f0)
    # receivers
    posrecs = zeros(2,3)
    posrecs[1,:] .= [lx/2,  2ly/3, lz/2]
    posrecs[2,:] .= [2lx/3, 2ly/3, lz/2]
    recs = Receivers(posrecs)

    solve3D(lx, ly, lz, lt, vel, srcs, recs;
            halo=0, do_vis=true, do_save=true, nvis=50, nsave=50, gif_name="acoustic3D_xPU_center_slice_halo0", save_name="acoustic3D_center_halo0", freetop=false, threshold=0.001, plims=[-1e-10,1e-10])
    solve3D(lx, ly, lz, lt, vel, srcs, recs;
            halo=20, rcoef=0.0001, do_vis=true, do_save=true, nvis=50, nsave=50, gif_name="acoustic3D_xPU_center_slice_halo20", save_name="acoustic3D_center_halo20", freetop=false, threshold=0.001, plims=[-1e-10,1e-10])
end

function run_gradient()
    # gradient velocity model
    # grid sizes
    nx, ny, nz = 121, 121, 81
    # model sizes [m]
    lx = 1000
    ly = 1000
    lz = 800
    vel = zeros(Float64, nx, ny, nz)
    for i=1:nx
        for j=1:ny
            for k = 1:nz
                vel[i,j,k] = 2000.0 + 12.0*(j-1)
            end
        end
    end
    # constant after some depth
    bigv = vel[1,ny-40,1]
    vel[:,ny-40:end,:] .= bigv

    lt = 2.0                                     # final time [s]
    # sources
    f0 = 15.0                                    # source dominating frequency [Hz]
    t0 = 4 / f0                                  # source activation time [s]
    stf = rickersource1D                         # second derivative of gaussian
    possrcs = zeros(4,3)
    possrcs[1,:] .= [lx/4, 100, lz/2]
    possrcs[2,:] .= [3lx/4, 100, lz/2]
    possrcs[3,:] .= [lx/2, 100, 2lz/5]
    possrcs[4,:] .= [lx/2, 100, 3lz/5]
    srcs = Sources(possrcs, fill(t0, 4), fill(stf, 4), f0)
    # receivers
    posrecs = zeros(2,3)
    posrecs[1,:] .= [lx/2,  2ly/3, lz/2]
    posrecs[2,:] .= [2lx/3, 2ly/3, lz/2]
    recs = Receivers(posrecs)

    solve3D(lx, ly, lz, lt, vel, srcs, recs;
            halo=20, rcoef=0.0001, do_vis=true, do_save=true, nvis=50, nsave=50, gif_name="acoustic3D_xPU_gradient_freetop_slice_halo20", save_name="acoustic3D_gradient_freetop_halo20", freetop=true, threshold=0.01, plims=[-1e-10,1e-10])
end

run_center()
run_gradient()