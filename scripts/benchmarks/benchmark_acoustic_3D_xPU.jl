include("../solvers/acoustic_3D_xPU.jl")

# benchmark single runs
nx = ny = nz = [8, 16, 32, 64, 128, 256, 320, 400, 451, 512, 550] .+ 1
lx = ly = lz = (nx .- 1) .* 10.0
for i = eachindex(nx)
    vel = 2000 .* ones(nx[i], ny[i], nz[i])
    possrcs = zeros(Int,1,3)
    possrcs[1,:] = [div(nx[i], 2, RoundUp), div(ny[i], 2, RoundUp), div(nz[i], 2, RoundUp)]
    acoustic3D_xPU(lx[i], ly[i], lz[i], 1, vel, possrcs; do_bench=true, do_vis=false, do_save=false, freetop=false)
end

# benchmark full run
t_tic = Base.time()
nx = ny = nz = 513
nt = 1000
lx = ly = lz = (nx - 1) * 10.0
vel = 2000 .* ones(nx, ny, nz)
possrcs = zeros(Int,1,3)
possrcs[1,:] = [div(nx, 2, RoundUp), div(ny, 2, RoundUp), div(nz, 2, RoundUp)]
acoustic3D_xPU(lx, ly, lz, nt, vel, possrcs; do_vis=false, do_save=false, freetop=false)
t_toc = Base.time() - t_tic
@printf("size = %dx%d, nt = %d, time = %1.3e sec\n", nx, ny, nt, t_toc)