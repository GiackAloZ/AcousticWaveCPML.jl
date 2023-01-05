using AcousticWaveCPML
using AcousticWaveCPML.Acoustic2D_CUDA

using Printf

# benchmark single runs
nx = ny = 2 .^ (6:14) .+ 1
lx = ly = (nx .- 1) .* 10.0
for i = eachindex(nx)
    vel = 2000.0 .* ones(Float64, nx[i], ny[i])   # velocity model [m/s]
    # sources
    f0 = 10.0                                     # source dominating frequency [Hz]
    t0 = 4 / f0                                   # source activation time [s]
    stf = rickersource1D                          # second derivative of gaussian
    possrcs = zeros(1,2)
    possrcs[1,:] .= [lx[i]/2, ly[i]/2]
    srcs = Sources(possrcs, [t0], [stf], f0)
    # receivers
    posrecs = zeros(2,2)
    posrecs[1,:] .= [lx[i]/2,  2ly[i]/3]
    recs = Receivers(posrecs)

    solve2D(lx[i], ly[i], 1.0, vel, srcs, recs; halo=20, rcoef=0.0001, do_bench=true, freetop=false)
end

# benchmark full run
t_tic = Base.time()
nx = ny = 2^14 + 1
lx = ly = (nx - 1) * 10.0
vel = 2000.0 .* ones(Float64, nx, ny)         # velocity model [m/s]
lt = 2.0                                      # final time [s]
# sources
f0 = 10.0                                     # source dominating frequency [Hz]
t0 = 4 / f0                                   # source activation time [s]
stf = rickersource1D                          # second derivative of gaussian
possrcs = zeros(1,2)
possrcs[1,:] .= [lx/2, ly/2]
srcs = Sources(possrcs, [t0], [stf], f0)
# receivers
posrecs = zeros(2,2)
posrecs[1,:] .= [lx/2,  2ly/3]
recs = Receivers(posrecs)

solve2D(lx, ly, lt, vel, srcs, recs; halo=20, rcoef=0.0001, freetop=false)

t_toc = Base.time() - t_tic
@printf("fullrun: size = %dx%d, nt = %d, time = %1.3e sec\n", nx, ny, size(recs.seismograms, 1), t_toc)