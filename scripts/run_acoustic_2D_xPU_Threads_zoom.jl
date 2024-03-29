using AcousticWaveCPML
using AcousticWaveCPML.Acoustic2D_Threads

using Logging
errorlogger = ConsoleLogger(stderr, Logging.Error)
global_logger(errorlogger)

include(joinpath(@__DIR__, "models/rescale_model.jl"))

function run_center()
    # simple constant velocity model
    nx = ny = 201                           # grid size
    lx = ly = 2000.0                        # model sizes [m]
    vel = 2000.0 .* ones(Float64, nx, ny)   # velocity model [m/s]
    lt = 20.0                                # final time [s]
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
            halo=5, rcoef=0.01, do_vis=true, nvis=100, gif_name="acoustic2D_xPU_center_halo5_zoom", freetop=false, threshold=0.001, plims=[-1e-10,1e-10])
    solve2D(lx, ly, lt, vel, srcs, recs;
            halo=10, rcoef=0.001, do_vis=true, nvis=100, gif_name="acoustic2D_xPU_center_halo10_zoom", freetop=false, threshold=0.001, plims=[-1e-10,1e-10])
    solve2D(lx, ly, lt, vel, srcs, recs;
            halo=20, rcoef=0.0001, do_vis=true, nvis=100, gif_name="acoustic2D_xPU_center_halo20_zoom", freetop=false, threshold=0.001, plims=[-1e-10,1e-10])

end

run_center()