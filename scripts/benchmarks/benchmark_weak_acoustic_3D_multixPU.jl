include("../solvers/acoustic_3D_multixPU.jl")

MPI.Init()

comm = MPI.COMM_WORLD
nprocs = MPI.Comm_size(comm)
dimsss = round(Int, cbrt(nprocs))

nx = ny = nz = 513
nt = 200

# simple constant velocity model
vel_func(x,y,z) = 2000.0
lx, ly, lz = ((nx-2)*dimsss + 2 - 1)*10.0, ((ny-2)*dimsss + 2 - 1)*10.0, ((nz-2)*dimsss + 2 - 1)*10.0
# one source in the center
possrcs = zeros(Int,1,3)
possrcs[1,:] = [div(nx, 2, RoundUp), div(ny, 2, RoundUp), div(nz, 2, RoundUp)]

acoustic3D_multixPU(lx, ly, lz, nx, ny, nz, nt, vel_func, possrcs;
                    halo=20, rcoef=0.0001, do_vis=false, do_save=false, freetop=false, init_MPI=false)

MPI.Finalize()