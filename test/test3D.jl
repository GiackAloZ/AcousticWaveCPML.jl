using Test, ReferenceTests

using AcousticWaveCPML

include("../scripts/solvers/acoustic_3D.jl")

REF_FLD = joinpath(@__DIR__, "references")

@testset "Test update_ψ!" begin
    nx = ny = nz = 101
    halo = 20
    pcur = ones(nx, ny, nz)
    _dx = _dy = _dz = 1 / 10.0
    pcur[[1,end],:,:] .= 0.0
    pcur[:,[1,end],:] .= 0.0
    pcur[:,:,[1,end]] .= 0.0

    ψ_x_l, ψ_x_r, ψ_y_l, ψ_y_r, ψ_z_l, ψ_z_r = ones(halo+1, ny, nz), ones(halo+1, ny, nz), ones(nx, halo+1, nz), ones(nx, halo+1, nz), ones(nx, ny, halo+1), ones(nx, ny, halo+1)
    a_x_hl, a_x_hr = 0.5 .* ones(halo+1), 2 .* ones(halo+1)
    b_K_x_hl, b_K_x_hr = 0.5 .* ones(halo+1), 2 .* ones(halo+1)
    a_y_hl, a_y_hr = 0.5 .* ones(halo+1), 2 .* ones(halo+1)
    b_K_y_hl, b_K_y_hr = 0.5 .* ones(halo+1), 2 .* ones(halo+1)
    a_z_hl, a_z_hr = 0.5 .* ones(halo+1), 2 .* ones(halo+1)
    b_K_z_hl, b_K_z_hr = 0.5 .* ones(halo+1), 2 .* ones(halo+1)

    update_ψ!(ψ_x_l, ψ_x_r, ψ_y_l, ψ_y_r, ψ_z_l, ψ_z_r, pcur,
              halo, _dx, _dy, _dz,
              a_x_hl, a_x_hr,
              b_K_x_hl, b_K_x_hr,
              a_y_hl, a_y_hr,
              b_K_y_hl, b_K_y_hr,
              a_z_hl, a_z_hr,
              b_K_z_hl, b_K_z_hr)
    
    a1 = 0.55 .* ones(ny,nz)
    a1[[1,end],:] .= 0.5
    a1[:,[1,end]] .= 0.5
    @test ψ_x_l[1,:,:] == a1
    @test ψ_x_l[2:end,:,:] == 0.5 .* ones(halo, ny, nz)
    a2 = 1.8 .* ones(ny,nz)
    a2[[1,end],:] .= 2.0
    a2[:,[1,end]] .= 2.0
    @test ψ_x_r[1:end-1,:,:] == 2.0 .* ones(halo, ny, nz)
    @test ψ_x_r[end,:,:] == a2

    a3 = 0.55 .* ones(nx,nz)
    a3[[1,end],:] .= 0.5
    a3[:,[1,end]] .= 0.5
    @test ψ_y_l[:,1,:] == a3
    @test ψ_y_l[:,2:end,:] == 0.5 .* ones(nx, halo, nz)
    a4 = 1.8 .* ones(nx,nz)
    a4[[1,end],:] .= 2.0
    a4[:,[1,end]] .= 2.0
    @test ψ_y_r[:,1:end-1,:] == 2.0 .* ones(nx, halo, nz)
    @test ψ_y_r[:,end,:] == a2

    a5 = 0.55 .* ones(ny,nz)
    a5[[1,end],:] .= 0.5
    a5[:,[1,end]] .= 0.5
    @test ψ_z_l[:,:,1] == a5
    @test ψ_z_l[:,:,2:end] == 0.5 .* ones(nx, ny, halo)
    a6 = 1.8 .* ones(nz,nz)
    a6[[1,end],:] .= 2.0
    a6[:,[1,end]] .= 2.0
    @test ψ_z_r[:,:,1:end-1] == 2.0 .* ones(nx, ny, halo)
    @test ψ_z_r[:,:,end] == a6
end

@testset "Reference test constant velocity" begin
    # simple constant velocity model
    nx = ny = nz = 61
    lx = ly = lz = (nx-1) * 10.0
    vel = 2000.0 .* ones(Float64, nx, ny, nz)
    nt = 1000
    # one source in the center
    possrcs = zeros(Int,1,3)
    possrcs[1,:] = [div(nx, 2, RoundUp), div(ny, 2, RoundUp), div(nz, 2, RoundUp)]

    pend = acoustic3D(lx, ly, lz, nt, vel, possrcs;
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
    nt = 1000
    # one source in the center
    possrcs = zeros(Int,1,3)
    possrcs[1,:] = [div(nx, 2, RoundUp), div(ny, 2, RoundUp), div(nz, 2, RoundUp)]

    pend = acoustic3D(lx, ly, lx, nt, vel, possrcs;
                      halo=20, rcoef=0.0001, do_vis=false, do_save=false, freetop=false)

    @test_reference joinpath(REF_FLD, "3D_gradient_center_halo20_nt1000.txt") pend
end
