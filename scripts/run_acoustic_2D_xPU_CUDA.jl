using AcousticWaveCPML
using AcousticWaveCPML.Acoustic2D_CUDA

using Logging
errorlogger = ConsoleLogger(stderr, Logging.Error)
global_logger(errorlogger)

include(joinpath(@__DIR__, "models/rescale_model.jl"))

function run_center()
    # simple constant velocity model
    nx = ny = 201                           # grid size
    lx = ly = 2000.0                        # model sizes [m]
    vel = 2000.0 .* ones(Float64, nx, ny)   # velocity model [m/s]
    lt = 2.0                                # final time [s]
    # sources
    f0 = 10.0                               # source dominating frequency [Hz]
    t0 = 4 / f0                             # source activation time [s]
    stf = rickersource1D                    # second derivative of gaussian
    possrcs = zeros(1,2)
    possrcs[1,:] .= [lx/2, ly/2]
    srcs = Sources(possrcs, [t0], [stf], f0)
    # receivers
    posrecs = zeros(2,2)
    posrecs[1,:] .= [lx/2,  2ly/3]
    posrecs[2,:] .= [2lx/3, 2ly/3]
    recs = Receivers(posrecs)

    solve2D(lx, ly, lt, vel, srcs, recs;
            halo=0, do_vis=true, nvis=20, gif_name="acoustic2D_xPU_center_halo0", freetop=false, threshold=0.001, plims=[-2e-8,2e-8])
    solve2D(lx, ly, lt, vel, srcs, recs;
            halo=5, rcoef=0.01, do_vis=true, nvis=20, gif_name="acoustic2D_xPU_center_halo5", freetop=false, threshold=0.001, plims=[-2e-8,2e-8])
    solve2D(lx, ly, lt, vel, srcs, recs;
            halo=10, rcoef=0.001, do_vis=true, nvis=20, gif_name="acoustic2D_xPU_center_halo10", freetop=false, threshold=0.001, plims=[-2e-8,2e-8])
    solve2D(lx, ly, lt, vel, srcs, recs;
            halo=20, rcoef=0.0001, do_vis=true, nvis=20, gif_name="acoustic2D_xPU_center_halo20", freetop=false, threshold=0.001, plims=[-2e-8,2e-8])

end

function run_center_freetop()
    # simple constant velocity model
    nx = ny = 201                           # grid size
    lx = ly = 2000.0                        # model sizes [m]
    vel = 2000.0 .* ones(Float64, nx, ny)   # velocity model [m/s]
    lt = 2.0                                # final time [s]
    # sources
    f0 = 10.0                               # source dominating frequency [Hz]
    t0 = 4 / f0                             # source activation time [s]
    stf = rickersource1D                    # second derivative of gaussian
    possrcs = zeros(1,2)
    possrcs[1,:] .= [lx/2, ly/2]
    srcs = Sources(possrcs, [t0], [stf], f0)
    # receivers
    posrecs = zeros(2,2)
    posrecs[1,:] .= [lx/2,  2ly/3]
    posrecs[2,:] .= [2lx/3, 2ly/3]
    recs = Receivers(posrecs)

    solve2D(lx, ly, lt, vel, srcs, recs;
            halo=0, do_vis=true, nvis=20, gif_name="acoustic2D_xPU_center_freetop_halo0", freetop=true, threshold=0.001, plims=[-2e-8,2e-8])
    solve2D(lx, ly, lt, vel, srcs, recs;
            halo=5, rcoef=0.01, do_vis=true, nvis=20, gif_name="acoustic2D_xPU_center_freetop_halo5", freetop=true, threshold=0.001, plims=[-2e-8,2e-8])
    solve2D(lx, ly, lt, vel, srcs, recs;
            halo=10, rcoef=0.001, do_vis=true, nvis=20, gif_name="acoustic2D_xPU_center_freetop_halo10", freetop=true, threshold=0.001, plims=[-2e-8,2e-8])
    solve2D(lx, ly, lt, vel, srcs, recs;
            halo=20, rcoef=0.0001, do_vis=true, nvis=20, gif_name="acoustic2D_xPU_center_freetop_halo20", freetop=true, threshold=0.001, plims=[-2e-8,2e-8])
end

function run_gradient()
    # gradient velocity model
    nx, ny = 211 * 2, 121 * 2
    vel = zeros(Float64, nx, ny);
    for i=1:nx
        for j=1:ny
            vel[i,j] = 2000.0 + 12.0*(j-1)
        end
    end
    # constant after some depth
    bigv = vel[1,ny-40]
    vel[:,ny-40:end] .= bigv
    lx = 2100.0 * 5
    ly = 1200.0 * 5
    lt = 5.0
    # sources
    f0 = 10.0                               # source dominating frequency [Hz]
    t0 = 4 / f0                             # source activation time [s]
    stf = rickersource1D                    # second derivative of gaussian
    possrcs = zeros(6,2)
    possrcs[1,:] .= [3lx/11, 5]
    possrcs[2,:] .= [4lx/11, 5]
    possrcs[3,:] .= [5lx/11, 5]
    possrcs[4,:] .= [6lx/11, 5]
    possrcs[5,:] .= [7lx/11, 5]
    possrcs[6,:] .= [8lx/11, 5]
    srcs = Sources(possrcs, fill(t0, 6), fill(stf, 6), f0)
    # receivers
    posrecs = zeros(2,2)
    posrecs[1,:] .= [lx/2,  2ly/3]
    posrecs[2,:] .= [2lx/3, 2ly/3]
    recs = Receivers(posrecs)

    solve2D(lx, ly, lt, vel, srcs, recs;
            halo=0, do_vis=true, nvis=20, gif_name="acoustic2D_xPU_gradient_freetop_halo0", freetop=true, threshold=0.01, plims=[-3e-9,3e-9])
    solve2D(lx, ly, lt, vel, srcs, recs;
            halo=5, rcoef=0.01, do_vis=true, nvis=20, gif_name="acoustic2D_xPU_gradient_freetop_halo5", freetop=true, threshold=0.01, plims=[-3e-9,3e-9])
    solve2D(lx, ly, lt, vel, srcs, recs;
            halo=10, rcoef=0.001, do_vis=true, nvis=20, gif_name="acoustic2D_xPU_gradient_freetop_halo10", freetop=true, threshold=0.01, plims=[-3e-9,3e-9])
    solve2D(lx, ly, lt, vel, srcs, recs;
            halo=20, rcoef=0.0001, do_vis=true, nvis=20, gif_name="acoustic2D_xPU_gradient_freetop_halo20", freetop=true, threshold=0.01, plims=[-3e-9,3e-9])
end

function run_complex()
    # complex velocity model (x5 resolution)
    nx, ny, nz = 143*5, 81*5, 70
    lx = 1430*10
    ly = 810*10
    lt = 7.0
    vel = permutedims(rescalemod(nz, nx, ny; kind="cubic"), [2, 3, 1])[:,:,div(nz,2)]

    # sources
    f0 = 10.0                               # source dominating frequency [Hz]
    t0 = 4 / f0                             # source activation time [s]
    stf = rickersource1D                    # second derivative of gaussian
    possrcs = zeros(6,2)
    possrcs[1,:] .= [3lx/11, 10]
    possrcs[2,:] .= [4lx/11, 10]
    possrcs[3,:] .= [5lx/11, 10]
    possrcs[4,:] .= [6lx/11, 10]
    possrcs[5,:] .= [7lx/11, 10]
    possrcs[6,:] .= [8lx/11, 10]
    srcs = Sources(possrcs, fill(t0, 6), fill(stf, 6), f0)
    # receivers
    posrecs = zeros(2,2)
    posrecs[1,:] .= [lx/2,  ly/2]
    posrecs[2,:] .= [4.5lx/11, 50]
    recs = Receivers(posrecs)

    solve2D(lx, ly, lt, vel, srcs, recs;
            halo=0, do_vis=true, nvis=50, gif_name="acoustic2D_xPU_complex_freetop_halo0", freetop=true, threshold=0.05, plims=[-2e-9,2e-9])
    solve2D(lx, ly, lt, vel, srcs, recs;
            halo=20, rcoef=0.0001, do_vis=true, nvis=50, gif_name="acoustic2D_xPU_complex_freetop_halo20", freetop=true, threshold=0.05, plims=[-2e-9,2e-9])
end

run_center()
run_center_freetop()
run_gradient()
run_complex()