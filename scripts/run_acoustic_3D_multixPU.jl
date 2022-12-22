include("solvers/acoustic_3D_multixPU.jl")
include("models/rescale_model.jl")

function run_simple()
    # simple constant velocity model (must be run with 8 MPI processes)
    nx = ny = nz = 52, 52, 52
    lx = ly = lz = ((nx-2)*2 + 2 - 1) * 10.0
    nt = 1000
    vel_func(x,y,z) = 2000.0

    # four sources on the top
    possrcs = zeros(Int,4,3)
    possrcs[1,:] = [div(102, 3, RoundUp), 3, div(101, 3, RoundUp)]
    possrcs[2,:] = [div(2 * 102, 3, RoundUp), 3, div(101, 3, RoundUp)]
    possrcs[3,:] = [div(102, 3, RoundUp), 3, div(2 * 101, 3, RoundUp)]
    possrcs[4,:] = [div(2 * 102, 3, RoundUp), 3, div(2 * 101, 3, RoundUp)]

    acoustic3D_multixPU(lx, ly, lz, nx, ny, nz, nt, vel_func, possrcs;
                        halo=20, rcoef=0.0001, do_vis=true, do_save=true, nsave=20,
                        gif_name="acoustic3Dmulti_four_halo20", save_name="acoustic3Dmulti_four_halo20", freetop=false, threshold=0.001, init_MPI=false)
end

function run_complex()
    # complex velocity model (x2 resolution, must be run with 8 MPI processes)
    nx, ny, nz = 143*2, 81*2, 70*2
    gnx, gny, gnz = (nx-2)*2+2, (ny-2)*2+2, (nz-2)*2+2
    lx = (gnx-1) * 10.0
    ly = (gny-1) * 10.0
    lz = (gnz-1) * 10.0
    nt = 8000
    vel = rescalemod(gnz, gnx, gny; kind="cubic", func=true)
    vel_func(x,y,z) = vel(
        round(Int, z / 10,RoundDown) * (70 - 1) / (gnz - 1) + 1,
        round(Int, x / 10,RoundDown) * (143 - 1) / (gnx - 1) + 1,
        round(Int, y / 10,RoundDown) * (81 - 1) / (gny - 1) + 1
        )

    # four sources on the top
    possrcs = zeros(Int,4,3)
    possrcs[1,:] = [div(gnx, 3, RoundUp), 3, div(gnz, 3, RoundUp)]
    possrcs[2,:] = [div(2gnx, 3, RoundUp), 3, div(gnz, 3, RoundUp)]
    possrcs[3,:] = [div(gnx, 3, RoundUp), 3, div(2gnz, 3, RoundUp)]
    possrcs[4,:] = [div(2gnx, 3, RoundUp), 3, div(2gnz, 3, RoundUp)]

    # run simulation
    acoustic3D_multixPU(lx, ly, lz, nx, ny, nz, nt, vel_func, possrcs;
            halo=20, rcoef=0.0001, do_vis=false, do_save=true, nsave=50, plims=[-1,1],
            gif_name="acoustic3Dmulti_complex_halo20_slice", save_name="acoustic3Dmulti_complex_halo20", freetop=true, threshold=0.01, init_MPI=false)
end

MPI.Init()

# run_simple()
run_complex()

MPI.Finalize()