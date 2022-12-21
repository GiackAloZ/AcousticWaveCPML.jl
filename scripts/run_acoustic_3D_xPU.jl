include("solvers/acoustic_3D_xPU.jl")
include("models/rescale_model.jl")
include("visualize3D.jl")

function run_simple()
    # simple constant velocity model
    nx, ny, nz = 102, 102, 101
    lx = (nx-1) * 10.0
    ly = (ny-1) * 10.0
    lz = (nz-1) * 10.0
    nt = 1000
    vel = 2000.0 .* ones(nx, ny, nz)

    # one source in the center
    possrcs = zeros(Int,1,3)
    possrcs[1,:] = [div(nx, 2, RoundUp), div(ny, 2, RoundUp), div(nz, 2, RoundUp)]

    # run simulation
    acoustic3D_xPU(lx, ly, lz, nt, vel, possrcs;
            halo=20, rcoef=0.0001, do_vis=true, do_save=true, nsave=20,
            gif_name="acoustic3D_center_halo20", save_name="acoustic3D_center_halo20", freetop=true, threshold=0.001)
    # generate pngs from saved results
    visualise_3D("acoustic3D_center_halo20"; frames=40:20:nt)
end

function run_gradient()
    # gradient velocity model
    nx, ny, nz = 102, 102, 101
    lx = (nx-1) * 10.0
    ly = (ny-1) * 10.0
    lz = (nz-1) * 10.0
    nt = 1000
    vel = zeros(Float64, nx, ny, nz)
    for i=1:nx
        for j=1:ny
            for k = 1:nz
                vel[i,j,k] = 2000.0 + 12.0*(j-1)
            end
        end
    end
    # constant after some depth
    bigv = vel[1,ny-40,1]
    vel[:,ny-40:end,:] .= bigv

    # four sources on the top
    possrcs = zeros(Int,4,3)
    possrcs[1,:] = [div(102, 3, RoundUp), 3, div(101, 3, RoundUp)]
    possrcs[2,:] = [div(2 * 102, 3, RoundUp), 3, div(101, 3, RoundUp)]
    possrcs[3,:] = [div(102, 3, RoundUp), 3, div(2 * 101, 3, RoundUp)]
    possrcs[4,:] = [div(2 * 102, 3, RoundUp), 3, div(2 * 101, 3, RoundUp)]

    # run simulation
    acoustic3D_xPU(lx, ly, lz, nt, vel, possrcs;
            halo=20, rcoef=0.0001, do_vis=true, do_save=true, nsave=20,
            gif_name="acoustic3D_gradient_halo20", save_name="acoustic3D_gradient_halo20", freetop=true, threshold=0.001)
    # generate pngs from saved results
    visualise_3D("acoustic3D_gradient_halo20"; frames=40:20:nt)
end

function run_complex()
    # complex velocity model
    nx, ny, nz = 70, 143, 81
    lx = (nx-1) * 10.0
    ly = (ny-1) * 10.0
    lz = (nz-1) * 10.0
    nt = 1000
    vel = permutedims(rescalemod(70, 143, 81, "cubic"), [2, 3, 1])

    # constant after some depth
    bigv = vel[1,ny-40,1]
    vel[:,ny-40:end,:] .= bigv

    # four sources on the top
    possrcs = zeros(Int,4,3)
    possrcs[1,:] = [div(102, 3, RoundUp), 3, div(101, 3, RoundUp)]
    possrcs[2,:] = [div(2 * 102, 3, RoundUp), 3, div(101, 3, RoundUp)]
    possrcs[3,:] = [div(102, 3, RoundUp), 3, div(2 * 101, 3, RoundUp)]
    possrcs[4,:] = [div(2 * 102, 3, RoundUp), 3, div(2 * 101, 3, RoundUp)]

    # run simulation
    acoustic3D_xPU(lx, ly, lz, nt, vel, possrcs;
            halo=20, rcoef=0.0001, do_vis=true, do_save=true, nsave=20,
            gif_name="acoustic3D_complex_halo20", save_name="acoustic3D_complex_halo20", freetop=true, threshold=0.001)
    # generate pngs from saved results
    visualise_3D("acoustic3D_complex_halo20"; frames=40:20:nt)
end

run_simple()
run_gradient()
run_complex()
