include("../solvers/acoustic_3D_xPU.jl")

# simple constant velocity model
nx = ny = nz = 513
nt = 200
vel = 2000.0 .* ones(Float64, nx, ny, nz)
lx, ly, lz = (nx-1)*10.0, (ny-1)*10.0, (nz-1)*10.0
# one source in the center
possrcs = zeros(Int,1,3)
possrcs[1,:] = [div(nx, 2, RoundUp), div(ny, 2, RoundUp), div(nz, 2, RoundUp)]

acoustic3D_xPU(lx, ly, lz, nt, vel, possrcs;
               halo=20, rcoef=0.0001, do_vis=false, do_save=false, do_bench=false, freetop=false)
