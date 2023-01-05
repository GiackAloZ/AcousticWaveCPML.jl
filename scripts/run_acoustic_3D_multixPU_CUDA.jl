using AcousticWaveCPML
using AcousticWaveCPML.Acoustic3Dmulti_CUDA

using Logging
errorlogger = ConsoleLogger(stderr, Logging.Error)
global_logger(errorlogger)

include(joinpath(@__DIR__, "models/rescale_model.jl"))

function run_complex()
    # complex velocity model (x2 resolution, must be run with 8 MPI processes to get x4 resolution)
    nx, ny, nz = 143*2, 81*2, 70*2                          # grid size
    gnx, gny, gnz = (nx-2)*2+2, (ny-2)*2+2, (nz-2)*2+2
    # model sizes [m]
    lx = (gnx-1) * 10.0
    ly = (gny-1) * 10.0
    lz = (gnz-1) * 10.0
    # final time [s]
    lt = 10.0
    vel = rescalemod(gnz, gnx, gny; kind="cubic", func=true)
    vel_func(x,y,z) = vel(
        round(Int, z / 10,RoundDown) * (70 - 1) / (gnz - 1) + 1,
        round(Int, x / 10,RoundDown) * (143 - 1) / (gnx - 1) + 1,
        round(Int, y / 10,RoundDown) * (81 - 1) / (gny - 1) + 1
        )

    # sources
    f0 = 10.0                                    # source dominating frequency [Hz]
    t0 = 4 / f0                                  # source activation time [s]
    stf = rickersource1D                         # second derivative of gaussian
    possrcs = zeros(4,3)
    possrcs[1,:] .= [lx/3, 30, lz/3]
    possrcs[2,:] .= [2lx/3, 30, lz/3]
    possrcs[3,:] .= [lx/3, 30, 2lz/3]
    possrcs[4,:] .= [2lx/3, 30, 2lz/3]
    srcs = Sources(possrcs, fill(t0, 4), fill(stf, 4), f0)
    # receivers
    posrecs = zeros(2,3)
    posrecs[1,:] .= [lx/2,  2ly/3, lz/2]
    posrecs[2,:] .= [2lx/3, 2ly/3, lz/2]
    recs = Receivers(posrecs)

    solve3D_multi(lx, ly, lz, lt, nx, ny, nz, vel_func, srcs, recs;
            halo=20, rcoef=0.0001, do_vis=false, do_save=true, nsave=50, save_name="acoustic3D_multixPU_complex_halo20", freetop=true, threshold=0.005, plims=[-1e-10,1e-10])
end

run_complex()