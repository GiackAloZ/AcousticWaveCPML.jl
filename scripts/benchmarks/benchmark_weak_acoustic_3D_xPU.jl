using AcousticWaveCPML
using AcousticWaveCPML.Acoustic3D_CUDA

nx = ny = nz = 513
# simple constant velocity model
vel = 2000.0 .* ones(Float64, nx, ny)
lx, ly = (nx-1)*10.0, (ny-1)*10.0
lt = 0.2
# sources
f0 = 10.0                                     # source dominating frequency [Hz]
t0 = 4 / f0                                   # source activation time [s]
stf = rickersource1D                          # second derivative of gaussian
possrcs = zeros(1,2)
possrcs[1,:] .= [lx/2, ly/2, lz/2]
srcs = Sources(possrcs, [t0], [stf], f0)
# receivers
posrecs = zeros(2,2)
posrecs[1,:] .= [lx/2,  2ly/3, lz/2]
recs = Receivers(posrecs)

solve3D(lx, ly, lz, lt, vel, srcs, recs;
              halo=20, rcoef=0.0001, freetop=false)
