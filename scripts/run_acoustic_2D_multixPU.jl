include("solvers/acoustic_2D_multixPU.jl")

# simple constant velocity model (must be run with 4 MPI processes)
nx, ny = 101, 101

vel_func(x,y) = 2000.0

# one source in the center
possrcs = zeros(Int,1,2)
possrcs[1,:] = [div(200, 2, RoundUp), div(200, 2, RoundUp)]

acoustic2D_multixPU(1990.0, 1990.0, nx, ny, 1000, vel_func, possrcs;
                    halo=20, rcoef=0.0001, do_vis=true,
                    gif_name="acoustic2Dmulti_center_halo20", freetop=false, threshold=0.001)