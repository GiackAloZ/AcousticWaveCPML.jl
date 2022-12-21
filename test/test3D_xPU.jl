using Test, ReferenceTests

using AcousticWaveCPML

include("../scripts/solvers/acoustic_3D_xPU.jl")

REF_FLD = joinpath(@__DIR__, "references")

@testset "Reference test constant velocity" begin
    # simple constant velocity model
    nx = ny = nz = 61
    lx = ly = lz = (nx-1) * 10.0
    vel = 2000.0 .* ones(Float64, nx, ny, nz)
    nt = 4000
    # one source in the center
    possrcs = zeros(Int,1,3)
    possrcs[1,:] = [div(nx, 2, RoundUp), div(ny, 2, RoundUp), div(nz, 2, RoundUp)]

    pend = acoustic3D_xPU(lx, ly, lz, nt, vel, possrcs;
                      halo=20, rcoef=0.0001, do_vis=false, do_save=false, freetop=false)

    @test_reference joinpath(REF_FLD, "3D_constant_center_halo20_nt1000.txt") pend
end

@testset "Reference test gradient velocity" begin
    # gradient velocity model
    nx = ny = nz = 61
    lx = ly = lz = (nx-1) * 10.0
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
    nt = 4000
    # one source in the center
    possrcs = zeros(Int,1,3)
    possrcs[1,:] = [div(nx, 2, RoundUp), div(ny, 2, RoundUp), div(nz, 2, RoundUp)]

    pend = acoustic3D_xPU(lx, ly, lx, nt, vel, possrcs;
                      halo=20, rcoef=0.0001, do_vis=false, do_save=false, freetop=false)

    @test_reference joinpath(REF_FLD, "3D_gradient_center_halo20_nt1000.txt") pend
end
