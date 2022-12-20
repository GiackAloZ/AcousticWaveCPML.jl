# to be run with only 1 MPI process

include("../solvers/acoustic_2D_multixPU.jl")

MPI.Init()

comm = MPI.COMM_WORLD
nprocs = MPI.Comm_size(comm)
dimsss = round(Int, sqrt(nprocs))

nx = ny = (2^14) + 1
nt = 1000

# simple constant velocity model
vel_func(x,y) = 2000.0
lx, ly = ((nx-2)*dimsss + 2 - 1)*10.0, ((ny-2)*dimsss + 2 - 1)*10.0
# one source in the center
possrcs = zeros(Int,1,2)
possrcs[1,:] = [div(nx, 2, RoundUp), div(ny, 2, RoundUp)]

acoustic2D_multixPU(lx, ly, nx, ny, nt, vel_func, possrcs;
                    halo=20, rcoef=0.0001, do_vis=false, freetop=false, init_MPI=false)

MPI.Finalize()