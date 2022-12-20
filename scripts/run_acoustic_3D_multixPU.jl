include("solvers/acoustic_3D_multixPU.jl")

# # simple constant velocity model (must be run with 3 MPI processes)
# nx, ny, nz = 52, 52, 101

# vel_func(x,y,z) = 2000.0

# # one source in the center
# possrcs = zeros(Int,1,3)
# possrcs[1,:] = [div(101, 2, RoundUp), div(101, 2, RoundUp), div(101, 2, RoundUp)]

# acoustic3D_multixPU(1010.0, 1010.0, 1000.0, nx, ny, nz, 1000, vel_func, possrcs;
#                     halo=20, rcoef=0.0001, do_vis=false, do_save=true, nsave=5,
#                     gif_name="acoustic3Dmulti_center_halo20", save_name="acoustic3Dmulti_center_halo20", freetop=false, threshold=0.001)

# simple constant velocity model (must be run with 3 MPI processes)
nx, ny, nz = 52, 52, 101

vel_func(x,y,z) = 2000.0

# four sources on the top
possrcs = zeros(Int,4,3)
possrcs[1,:] = [div(102, 3, RoundUp), 3, div(101, 3, RoundUp)]
possrcs[2,:] = [div(2 * 102, 3, RoundUp), 3, div(101, 3, RoundUp)]
possrcs[3,:] = [div(102, 3, RoundUp), 3, div(2 * 101, 3, RoundUp)]
possrcs[4,:] = [div(2 * 102, 3, RoundUp), 3, div(2 * 101, 3, RoundUp)]

acoustic3D_multixPU(1010.0, 1010.0, 1000.0, nx, ny, nz, 1000, vel_func, possrcs;
                    halo=20, rcoef=0.0001, do_vis=true, do_save=true, nsave=20,
                    gif_name="acoustic3Dmulti_four_halo20", save_name="acoustic3Dmulti_four_halo20", freetop=true, threshold=0.001)