using AcousticWaveCPML
using AcousticWaveCPML.Acoustic3Dmulti_CUDA

import MPI

MPI.Init()

comm = MPI.COMM_WORLD
nprocs = MPI.Comm_size(comm)
dimsss = round(Int, sqrt(nprocs))

nx = ny = nz = 513
# simple constant velocity model
vel_func(x,y,z) = 2000.0
lx, ly, lz = ((nx-2)*dimsss + 2 - 1)*10.0, ((ny-2)*dimsss + 2 - 1)*10.0, ((nz-2)*dimsss + 2 - 1)*10.0
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

solve3D_multi(lx, ly, lz, lt, nx, ny, nz, vel_func, srcs, recs;
              halo=20, rcoef=0.0001, freetop=false, init_MPI=false)

MPI.Finalize()