include("../solvers/acoustic_2D_xPU.jl")

# benchmark single runs
nx = ny = 2 .^ (5:12) .+ 1
lx = ly = (nx .- 1) .* 10.0
for i = eachindex(nx)
    vel = 2000 .* ones(nx[i], ny[i])
    possrcs = zeros(Int,1,2)
    possrcs[1,:] = [div(nx[i], 2, RoundUp), div(ny[i], 2, RoundUp)]
    acoustic2D_xPU(lx[i], ly[i], 1, vel, possrcs; do_bench=true, freetop=false)
end

# benchmark full run
t_tic = Base.time()
nx = ny = 129
nt = 1000
lx = ly = (nx - 1) * 10.0
vel = 2000 .* ones(nx, ny)
possrcs = zeros(Int,1,2)
possrcs[1,:] = [div(nx, 2, RoundUp), div(ny, 2, RoundUp)]
acoustic2D_xPU(lx, ly, nt, vel, possrcs; do_vis=false, freetop=false)
t_toc = Base.time() - t_tic
@printf("size = %dx%d, nt = %d, time = %1.3e sec\n", nx, ny, nt, t_toc)