include("../solvers/acoustic_2D_xPU.jl")

nx = ny = (2^14) + 1
lx = ly = (nx - 1) * 10.0
nt = 1000
vel = 2000 .* ones(nx, ny)
possrcs = zeros(Int,1,2)
possrcs[1,:] = [div(nx, 2, RoundUp), div(ny, 2, RoundUp)]
acoustic2D_xPU(lx, ly, nt, vel, possrcs; do_bench=false, do_vis=false, freetop=false)
