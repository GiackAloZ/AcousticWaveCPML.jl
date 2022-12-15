include("solvers/acoustic_3D_xPU.jl")

# simple constant velocity model
nx, ny, nz = 101, 101, 101
vel = 2000.0 .* ones(Float64, nx, ny, nz);
# one source in the center
possrcs = zeros(Int,1,3)
possrcs[1,:] = [div(nx, 2, RoundUp), div(ny, 2, RoundUp), div(nz, 2, RoundUp)]

acoustic3D_xPU(1000.0, 1000.0, 1000.0, 1000, vel, possrcs; halo=5, rcoef=0.01, do_vis=true, gif_name="acoustic3D_center_halo5", save_name="acoustic3D_center_halo5", freetop=false, threshold=0.001)
acoustic3D_xPU(1000.0, 1000.0, 1000.0, 1000, vel, possrcs; halo=10, rcoef=0.001, do_vis=true, gif_name="acoustic3D_center_halo10", save_name="acoustic3D_center_halo10", freetop=false, threshold=0.001)
acoustic3D_xPU(1000.0, 1000.0, 1000.0, 1000, vel, possrcs; halo=20, rcoef=0.0001, do_vis=true, gif_name="acoustic3D_center_halo20", save_name="acoustic3D_center_halo20", freetop=false, threshold=0.001)
acoustic3D_xPU(1000.0, 1000.0, 1000.0, 1000, vel, possrcs; halo=40, rcoef=0.00001, do_vis=true, gif_name="acoustic3D_center_halo40", save_name="acoustic3D_center_halo40", freetop=false, threshold=0.001)
