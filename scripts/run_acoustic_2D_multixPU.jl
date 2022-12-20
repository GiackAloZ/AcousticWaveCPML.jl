include("solvers/acoustic_2D_multixPU.jl")

MPI.Init()

comm = MPI.COMM_WORLD
nprocs = MPI.Comm_size(comm)
dimsss = round(Int, sqrt(nprocs))

nx, ny = 211, 121
gnx, gny = (nx-2)*dimsss + 2, (ny-2)*dimsss + 2
lx, ly = ((nx-2)*dimsss + 2 - 1)*10.0, ((ny-2)*dimsss + 2 - 1)*10.0
nt = 1000

# velocity gradient
function vel_func(x,y)
    if y / 10.0 >= 430
        return 2000.0 + 12.0 * 430
    end
    return 2000.0 + 12.0 * (y / 10.0)
end

# one source in the center
possrcs = zeros(Int,6,2)
possrcs[1,:] = [div(3gnx, 11, RoundUp), 3]
possrcs[2,:] = [div(4gnx, 11, RoundUp), 3]
possrcs[3,:] = [div(5gnx, 11, RoundUp), 3]
possrcs[4,:] = [div(6gnx, 11, RoundUp), 3]
possrcs[5,:] = [div(7gnx, 11, RoundUp), 3]
possrcs[6,:] = [div(8gnx, 11, RoundUp), 3]

acoustic2D_multixPU(lx, ly, nx, ny, nt, vel_func, possrcs;
                    halo=20, rcoef=0.0001, do_vis=true,
                    gif_name="acoustic2Dmulti_big_halo20", freetop=true, threshold=0.01, init_MPI=false)

MPI.Finalize()