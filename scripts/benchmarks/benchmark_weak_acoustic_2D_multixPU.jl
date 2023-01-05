using AcousticWaveCPML
using AcousticWaveCPML.Acoustic2Dmulti_CUDA

import MPI

MPI.Init()

comm = MPI.COMM_WORLD
nprocs = MPI.Comm_size(comm)
dimsss = round(Int, sqrt(nprocs))

nx = ny = (2^14) + 1

# simple constant velocity model
vel_func(x,y) = 2000.0
lx, ly = ((nx-2)*dimsss + 2 - 1)*10.0, ((ny-2)*dimsss + 2 - 1)*10.0
lt = 1.0
# sources
f0 = 10.0                                     # source dominating frequency [Hz]
t0 = 4 / f0                                   # source activation time [s]
stf = rickersource1D                          # second derivative of gaussian
possrcs = zeros(1,2)
possrcs[1,:] .= [lx/2, ly/2]
srcs = Sources(possrcs, [t0], [stf], f0)
# receivers
posrecs = zeros(1,2)
posrecs[1,:] .= [lx/2,  2ly/3]
recs = Receivers(posrecs)

solve2D_multi(lx, ly, lt, nx, ny, vel_func, srcs, recs;
              halo=20, rcoef=0.0001, freetop=false, init_MPI=false)

MPI.Finalize()