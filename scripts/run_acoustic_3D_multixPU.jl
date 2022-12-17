include("solvers/acoustic_3D_multixPU.jl")

# simple constant velocity model
nx, ny, nz = 52, 52, 101

vel_func(x,y,z) = 2000.0

# one source in the center
possrcs = zeros(Int,1,3)
possrcs[1,:] = [div(101, 2, RoundUp), div(101, 2, RoundUp), div(101, 2, RoundUp)]

acoustic3D_multixPU(1000.0, 1000.0, 1000.0, nx, ny, nz, 1000, vel_func, possrcs; halo=20, rcoef=0.0001, do_vis=false, do_save=true, save_name="acoustic3D_center_halo20", freetop=false, threshold=0.001)