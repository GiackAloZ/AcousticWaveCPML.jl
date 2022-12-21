include("solvers/acoustic_2D_xPU.jl")

function run_simple()
    # simple constant velocity model
    nx, ny = 211, 211
    vel = 2000.0 .* ones(Float64, nx, ny)
    lx = (nx-1) * 10.0
    ly = (ny-2) * 10.0
    nt = 1000
    # one source in the center
    possrcs = zeros(Int,1,2)
    possrcs[1,:] = [div(nx, 2, RoundUp), div(ny, 2, RoundUp)]

    acoustic2D_xPU(lx, ly, nt, vel, possrcs; halo=0, do_vis=true, gif_name="acoustic2D_xPU_center_halo0", freetop=false, threshold=0.001)
    acoustic2D_xPU(lx, ly, nt, vel, possrcs; halo=5, rcoef=0.01, do_vis=true, gif_name="acoustic2D_xPU_center_halo5", freetop=false, threshold=0.001)
    acoustic2D_xPU(lx, ly, nt, vel, possrcs; halo=10, rcoef=0.001, do_vis=true, gif_name="acoustic2D_xPU_center_halo10", freetop=false, threshold=0.001)
    acoustic2D_xPU(lx, ly, nt, vel, possrcs; halo=20, rcoef=0.0001, do_vis=true, gif_name="acoustic2D_xPU_center_halo20", freetop=false, threshold=0.001)

end

function run_gradient_multi()
    # gradient velocity model
    nx, ny = 211, 121
    vel = zeros(Float64, nx, ny);
    for i=1:nx
        for j=1:ny
            vel[i,j] = 2000.0 + 12.0*(j-1)
        end
    end
    # constant after some depth
    bigv = vel[1,ny-40]
    vel[:,ny-40:end] .= bigv
    lx = (nx-1) * 10.0
    ly = (ny-2) * 10.0
    nt = 1500

    # 6 equidistant sources on top
    possrcs = zeros(Int,6,2)
    possrcs[1,:] = [div(3nx, 11, RoundUp), 3]
    possrcs[2,:] = [div(4nx, 11, RoundUp), 3]
    possrcs[3,:] = [div(5nx, 11, RoundUp), 3]
    possrcs[4,:] = [div(6nx, 11, RoundUp), 3]
    possrcs[5,:] = [div(7nx, 11, RoundUp), 3]
    possrcs[6,:] = [div(8nx, 11, RoundUp), 3]

    acoustic2D_xPU(lx, ly, nt, vel, possrcs; halo=0, do_vis=true, gif_name="acoustic2D_xPU_gradient_halo0")
    acoustic2D_xPU(lx, ly, nt, vel, possrcs; halo=5, rcoef=0.01, do_vis=true, gif_name="acoustic2D_xPU_gradient_halo5")
    acoustic2D_xPU(lx, ly, nt, vel, possrcs; halo=10, rcoef=0.001, do_vis=true, gif_name="acoustic2D_xPU_gradient_halo10")
    acoustic2D_xPU(lx, ly, nt, vel, possrcs; halo=20, rcoef=0.0001, do_vis=true, gif_name="acoustic2D_xPU_gradient_halo20")
    
end

function run_complex()
    # complex velocity model (x10 resolution)
    nx, ny, nz = 70*10, 143*10, 81*10
    lx = (nx-1) * 10.0
    ly = (ny-1) * 10.0
    nt = 8000
    vel = permutedims(rescalemod(nx, ny, nz, "cubic"), [2, 3, 1])[:,:,div(nz,2)]

    # 6 equidistant sources on top
    possrcs = zeros(Int,6,2)
    possrcs[1,:] = [div(3nx, 11, RoundUp), 3]
    possrcs[2,:] = [div(4nx, 11, RoundUp), 3]
    possrcs[3,:] = [div(5nx, 11, RoundUp), 3]
    possrcs[4,:] = [div(6nx, 11, RoundUp), 3]
    possrcs[5,:] = [div(7nx, 11, RoundUp), 3]
    possrcs[6,:] = [div(8nx, 11, RoundUp), 3]

    # run simulation
    acoustic2D_xPU(lx, ly, nt, vel, possrcs;
                   halo=20, rcoef=0.0001, do_vis=true,
                   gif_name="acoustic2D_xPU_complex_halo20", freetop=true, threshold=0.001)

end

run_simple()
run_gradient_multi()
run_complex()
