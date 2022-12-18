# to be run with only 1 MPI process

include("../solvers/acoustic_3D_multixPU.jl")

MPI.Init()

nxs = nys = nzs = (2 .^ (5:14)) .+ 1
nts     = round.(Int, [1e5, 1e5, 1e5, 1e5, 1e5, 1e4, 1e4, 1e3, 1e3, 1e3])
vel_func(x,y) = 2000.0
for (nx, ny, nt) in zip(nxs, nys, nts)
    # simple constant velocity model (must be run with 1 MPI process)
    lx, ly = ((nx-2)*1 + 2 - 1)*10.0, ((ny-2)*1 + 2 - 1)*10.0
    # one source in the center
    possrcs = zeros(Int,1,2)
    possrcs[1,:] = [div(nx, 2, RoundUp), div(ny, 2, RoundUp)]

    acoustic3D_multixPU(lx, ly, lz, nx, ny, nz, nt, vel_func, possrcs;
                        halo=20, rcoef=0.0001, do_vis=false, freetop=false, init_MPI=false)
end

MPI.Finalize()