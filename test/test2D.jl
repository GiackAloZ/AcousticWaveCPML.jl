using Test, ReferenceTests

using AcousticWaveCPML

include("../scripts/solvers/acoustic_2D.jl")

REF_FLD = joinpath(@__DIR__, "references")

@testset "Test update_ψ!" begin
    nx = ny = 101
    halo = 20
    pcur = ones(nx, ny)
    _dx = _dy = 1 / 10.0
    pcur[[1,end],:] .= 0.0
    pcur[:,[1,end]] .= 0.0

    ψ_x_l, ψ_x_r, ψ_y_l, ψ_y_r = ones(halo+1, ny), ones(halo+1, ny), ones(nx, halo+1), ones(nx, halo+1)
    a_x_hl, a_x_hr = 0.5 .* ones(halo+1), 2 .* ones(halo+1)
    b_K_x_hl, b_K_x_hr = 0.5 .* ones(halo+1), 2 .* ones(halo+1)
    a_y_hl, a_y_hr = 0.5 .* ones(halo+1), 2 .* ones(halo+1)
    b_K_y_hl, b_K_y_hr = 0.5 .* ones(halo+1), 2 .* ones(halo+1)

    update_ψ!(ψ_x_l, ψ_x_r, ψ_y_l, ψ_y_r, pcur,
              halo, _dx, _dy,
              a_x_hl, a_x_hr,
              b_K_x_hl, b_K_x_hr,
              a_y_hl, a_y_hr,
              b_K_y_hl, b_K_y_hr)
    
    @test ψ_x_l[1,:] == [0.5, fill(0.55, ny-2)..., 0.5]
    @test ψ_x_l[2:end,:] == 0.5 .* ones(halo, ny)
    @test ψ_x_r[1:end-1,:] == 2.0 .* ones(halo, ny)
    @test ψ_x_r[end,:] == [2.0, fill(1.8, ny-2)..., 2.0]

    @test ψ_y_l[:,1] == [0.5, fill(0.55, nx-2)..., 0.5]
    @test ψ_y_l[:,2:end] == 0.5 .* ones(nx, halo)
    @test ψ_y_r[:,1:end-1] == 2.0 .* ones(nx, halo)
    @test ψ_y_r[:,end] == [2.0, fill(1.8, nx-2)..., 2.0]
end

@testset "Reference test constant velocity" begin
    # simple constant velocity model
    nx = ny = 201
    lx = ly = (nx-1) * 10.0
    vel = 2000.0 .* ones(Float64, nx, ny)
    nt = 1000
    # one source in the center
    possrcs = zeros(Int,1,2)
    possrcs[1,:] = [div(nx, 2, RoundUp), div(ny, 2, RoundUp)]

    pend = acoustic2D(lx, ly, nt, vel, possrcs;
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

    pend = acoustic2D(lx, ly, nt, vel, possrcs;
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

    pend = acoustic2D(lx, ly, nt, vel, possrcs;
                      halo=20, rcoef=0.0001, do_vis=false, freetop=true)

    @test_reference joinpath(REF_FLD, "2D_gradient_multi_freetop_halo20_nt1000.txt") pend
end