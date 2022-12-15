include("../solvers/acoustic_2D.jl")

# benchmark
nx = ny = 2 .^ (5:12) .+ 1
lx = ly = (nx .- 1) .* 10.0
for i = eachindex(nx)
    vel = 2000 .* ones(nx[i], ny[i])
    possrcs = zeros(Int,1,2)
    possrcs[1,:] = [div(nx[i], 2, RoundUp), div(ny[i], 2, RoundUp)]
    acoustic2D(lx[i], ly[i], 1, vel, possrcs; do_bench=true, freetop=false)
end