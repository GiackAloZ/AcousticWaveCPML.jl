using AcousticWaveCPML
using AcousticWaveCPML.Acoustic3D

using Printf

# benchmark single runs
nx = ny = nz = [64, 128, 256, 320, 400] .+ 1
lx = ly = lz = (nx .- 1) .* 10.0
for i = eachindex(nx)
    vel = 2000.0 .* ones(Float64, nx[i], ny[i], nz[i])   # velocity model [m/s]
    # sources
    f0 = 10.0                                     # source dominating frequency [Hz]
    t0 = 4 / f0                                   # source activation time [s]
    stf = rickersource1D                          # second derivative of gaussian
    possrcs = zeros(1,3)
    possrcs[1,:] .= [lx[i]/2, ly[i]/2, lz[i]/2]
    srcs = Sources(possrcs, [t0], [stf], f0)
    # receivers
    posrecs = zeros(1,3)
    posrecs[1,:] .= [lx[i]/2,  2ly[i]/3, lz[i]/2]
    recs = Receivers(posrecs)

    solve3D(lx[i], ly[i], lz[i], 1.0, vel, srcs, recs; halo=20, rcoef=0.0001, do_bench=true, freetop=false)
end

# benchmark full run
t_tic = Base.time()
nx = ny = nz = 400 + 1
lx = ly = lz = (nx - 1) * 10.0
vel = 2000.0 .* ones(Float64, nx, ny, nz)     # velocity model [m/s]
lt = 0.1                                      # final time [s]
# sources
f0 = 10.0                                     # source dominating frequency [Hz]
t0 = 4 / f0                                   # source activation time [s]
stf = rickersource1D                          # second derivative of gaussian
possrcs = zeros(1,3)
possrcs[1,:] .= [lx/2, ly/2, lz/2]
srcs = Sources(possrcs, [t0], [stf], f0)
# receivers
posrecs = zeros(1,3)
posrecs[1,:] .= [lx/2,  2ly/3, lz/2]
recs = Receivers(posrecs)

solve3D(lx, ly, lz, lt, vel, srcs, recs; halo=20, rcoef=0.0001, freetop=false)

t_toc = Base.time() - t_tic
@printf("fullrun: size = %dx%dx%d, nt = %d, time = %1.3e sec\n", nx, ny, nz, size(recs.seismograms, 1), t_toc)