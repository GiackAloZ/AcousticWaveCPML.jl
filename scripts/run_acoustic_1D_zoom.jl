using AcousticWaveCPML
using AcousticWaveCPML.Acoustic1D

function run_center()
    # simple constant velocity model
    nx = 201                                # grid size
    lx = 2000.0                             # model sizes [m]
    vel = 2000.0 .* ones(Float64, nx)       # velocity model [m/s]
    lt = 10.0                                # final time [s]
    # sources
    f0 = 10.0                               # source dominating frequency [Hz]
    t0 = 4 / f0                             # source activation time [s]
    stf = rickersource1D                    # second derivative of gaussian
    possrcs = zeros(1,1)
    possrcs[1,:] .= [lx/2]
    srcs = Sources(possrcs, [t0], [stf], f0)
    # receivers
    posrecs = zeros(1,1)
    posrecs[1,:] .= [2lx/3]
    recs = Receivers(posrecs)

    solve1D(lx, lt, vel, srcs, recs;
            halo=5, rcoef=0.01, do_vis=true, nvis=50, gif_name="acoustic1D_center_halo5_zoom", plims=[-1e-7,1e-7])
    solve1D(lx, lt, vel, srcs, recs;
            halo=10, rcoef=0.001, do_vis=true, nvis=50, gif_name="acoustic1D_center_halo10_zoom", plims=[-1e-7,1e-7])
    solve1D(lx, lt, vel, srcs, recs;
            halo=20, rcoef=0.0001, do_vis=true, nvis=50, gif_name="acoustic1D_center_halo20_zoom", plims=[-1e-7,1e-7])

end

run_center()