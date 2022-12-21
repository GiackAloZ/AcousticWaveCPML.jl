using Test, ReferenceTests

using AcousticWaveCPML

include("../scripts/solvers/acoustic_2D_xPU.jl")

REF_FLD = joinpath(@__DIR__, "references")

@testset "Reference test constant velocity" begin
    # simple constant velocity model
    nx = ny = 201
    lx = ly = (nx-1) * 10.0
    vel = 2000.0 .* ones(Float64, nx, ny)
    nt = 1000
    # one source in the center
    possrcs = zeros(Int,1,2)
    possrcs[1,:] = [div(nx, 2, RoundUp), div(ny, 2, RoundUp)]

    pend = acoustic2D_xPU(lx, ly, nt, vel, possrcs;
                      halo=20, rcoef=0.0001, do_vis=false, freetop=false)

    @test_reference joinpath(REF_FLD, "2D_constant_center_halo20_nt1000.txt") pend
end

@testset "Reference test gradient velocity" begin
    # gradient velocity model
    nx = ny = 201
    lx = ly = (nx-1) * 10.0
    vel = zeros(Float64, nx, ny);
    for i=1:nx
        for j=1:ny
            vel[i,j] = 2000.0 + 12.0*(j-1)
        end
    end
    # constant after some depth
    bigv = vel[1,ny-40]
    vel[:,ny-40:end] .= bigv
    nt = 1000
    # one source in the center
    possrcs = zeros(Int,1,2)
    possrcs[1,:] = [div(nx, 2, RoundUp), div(ny, 2, RoundUp)]

    pend = acoustic2D_xPU(lx, ly, nt, vel, possrcs;
                      halo=20, rcoef=0.0001, do_vis=false, freetop=false)

    @test_reference joinpath(REF_FLD, "2D_gradient_center_halo20_nt1000.txt") pend
end

@testset "Reference test gradient velocity multi-source free top" begin
    # gradient velocity model
    nx = ny = 201
    lx = ly = (nx-1) * 10.0
    vel = zeros(Float64, nx, ny);
    for i=1:nx
        for j=1:ny
            vel[i,j] = 2000.0 + 12.0*(j-1)
        end
    end
    # constant after some depth
    bigv = vel[1,ny-40]
    vel[:,ny-40:end] .= bigv
    nt = 1000
    # 6 equidistant sources on top
    possrcs = zeros(Int,6,2)
    possrcs[1,:] = [div(3nx, 11, RoundUp), 3]
    possrcs[2,:] = [div(4nx, 11, RoundUp), 3]
    possrcs[3,:] = [div(5nx, 11, RoundUp), 3]
    possrcs[4,:] = [div(6nx, 11, RoundUp), 3]
    possrcs[5,:] = [div(7nx, 11, RoundUp), 3]
    possrcs[6,:] = [div(8nx, 11, RoundUp), 3]

    pend = acoustic2D_xPU(lx, ly, nt, vel, possrcs;
                      halo=20, rcoef=0.0001, do_vis=false, freetop=true)

    @test_reference joinpath(REF_FLD, "2D_gradient_multi_freetop_halo20_nt1000.txt") pend
end