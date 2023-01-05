using AcousticWaveCPML
using AcousticWaveCPML.Acoustic2Dmulti_Threads

using Logging
errorlogger = ConsoleLogger(stderr, Logging.Error)
global_logger(errorlogger)

import MPI
MPI.Init()

function run_center()
    # simple constant velocity model
    nx = ny = 201                           # grid size
    lx = ly = 2000.0                        # model sizes [m]
    vel_func(x,y) = 2000.0                  # velocity model [m/s]
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

    solve2D_multi(lx, ly, lt, nx, ny, vel_func, srcs, recs;
            halo=0, do_vis=true, nvis=20, gif_name="acoustic2D_multixPU_center_halo0", freetop=false, threshold=0.001, plims=[-2e-8,2e-8], init_MPI=false)
    solve2D_multi(lx, ly, lt, nx, ny, vel_func, srcs, recs;
            halo=5, rcoef=0.01, do_vis=true, nvis=20, gif_name="acoustic2D_multixPU_center_halo5", freetop=false, threshold=0.001, plims=[-2e-8,2e-8], init_MPI=false)
    solve2D_multi(lx, ly, lt, nx, ny, vel_func, srcs, recs;
            halo=10, rcoef=0.001, do_vis=true, nvis=20, gif_name="acoustic2D_multixPU_center_halo10", freetop=false, threshold=0.001, plims=[-2e-8,2e-8], init_MPI=false)
    solve2D_multi(lx, ly, lt, nx, ny, vel_func, srcs, recs;
            halo=20, rcoef=0.0001, do_vis=true, nvis=20, gif_name="acoustic2D_multixPU_center_halo20", freetop=false, threshold=0.001, plims=[-2e-8,2e-8], init_MPI=false)

end

run_center()

MPI.Finalize()
