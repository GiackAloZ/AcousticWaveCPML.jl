# to be run with only 1 MPI process

include("../solvers/acoustic_3D_multixPU.jl")

MPI.Init()

nxs = nys = nzs = [32, 64, 128, 256, 320, 400, 450, 512, 550, 600, 650] .+ 1
nts     = round.(Int, [1e3, 1e3, 1e3, 1e2, 1e2, 1e2, 1e2, 1e2, 1e2, 1e2, 1e2])
vel_func(x,y,z) = 2000.0
for (nx, ny, nz, nt) in zip(nxs, nys, nzs, nts)
    # simple constant velocity model (must be run with 1 MPI process)
    lx, ly, lz = ((nx-2)*1 + 2 - 1)*10.0, ((ny-2)*1 + 2 - 1)*10.0, ((nz-2)*1 + 2 - 1)*10.0
    # one source in the center
    possrcs = zeros(Int,1,3)
    possrcs[1,:] = [div(nx, 2, RoundUp), div(ny, 2, RoundUp), div(nz, 2, RoundUp)]

    acoustic3D_multixPU(lx, ly, lz, nx, ny, nz, nt, vel_func, possrcs;
                        halo=20, rcoef=0.0001, do_vis=false, do_save=false, freetop=false, init_MPI=false)
end

MPI.Finalize()